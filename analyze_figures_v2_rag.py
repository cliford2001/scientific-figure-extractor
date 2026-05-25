"""
LLM Analyzer for Extracted Figures — v2 RAG
=============================================

Analiza figuras Y tablas científicas con dos modos de contexto:

  context_strategy="full"  (por defecto)
      Inyecta el texto COMPLETO del paper como contexto, truncado solo si
      supera MAX_CONTEXT_WORDS (0 = sin límite). Ideal para GPUs con ctx
      grande (≥ 32 768 tokens).

  context_strategy="bm25"
      Recupera los top_k chunks más relevantes por figura/tabla usando BM25
      (rank_bm25 o fallback TF-IDF). Mejor para GPUs pequeñas o papers muy
      largos.

En ambos modos:
  - El abstract completo (o primeras ABSTRACT_MAX_WORDS palabras) se inyecta
    en los prompts de INFERENCIA pura también.
  - Las tablas y figuras comparten el mismo pipeline (kind="table" activa
    los templates especializados).

Uso:
    python analyze_figures_v2_rag.py extracted/figures.json --pdf paper.pdf
    python analyze_figures_v2_rag.py figures.json --context-file paper.txt --context-strategy full
    python analyze_figures_v2_rag.py figures.json --pdf paper.pdf --context-strategy bm25 --top-k 8
    python analyze_figures_v2_rag.py figures.json --pdf paper.pdf --max-context-words 6000

Requiere (para estrategia BM25):
    pip install rank-bm25
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests


# ─── Defaults ─────────────────────────────────────────────────────────────────
# Estos parámetros se pueden sobreescribir por CLI o por run_analysis_batch.py.
# Ajustar según la GPU disponible:
#   GTX 1080 Ti (11 GB, ctx 12 288) → MAX_CONTEXT_WORDS ≈ 3000–4000
#   RTX 3090 / A100  (ctx 32 768)   → MAX_CONTEXT_WORDS = 0 (sin límite)

DEFAULT_SERVER            = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MAX_TOKENS        = 1500       # tokens de respuesta del LLM
DEFAULT_TEMPERATURE       = 0.0        # determinista; evita alucinaciones
DEFAULT_TIMEOUT           = 300        # segundos por llamada
DEFAULT_CONTEXT_STRATEGY  = "full"     # "full" | "bm25"
DEFAULT_MAX_CONTEXT_WORDS = 0          # 0 = sin límite; >0 = truncar texto completo
DEFAULT_ABSTRACT_WORDS    = 0          # 0 = abstract completo detectado por sección
DEFAULT_CHUNK_WORDS       = 250        # tamaño de chunk en modo bm25
DEFAULT_CHUNK_OVERLAP     = 40         # overlap entre chunks en modo bm25
DEFAULT_TOP_K             = 10         # chunks BM25 recuperados por ítem (Lewis et al. 2020 usa 5; subimos a 10 por ventanas mayores)
MAX_RETRIES               = 5
RETRY_BACKOFF             = 3

# ─── Configuración recomendada por GPU ────────────────────────────────────────
#
#  GTX 1080 Ti  (11 GB, ctx 12 288):
#    --context-strategy full --max-context-words 3000
#
#  RTX 3090 / RTX 4090  (ctx 32 768):
#    --context-strategy full --max-context-words 0
#
#  A100 / H100  (ctx 128 000+):
#    --context-strategy full --max-context-words 0
#
#  Papers muy largos en GPU chica:
#    --context-strategy bm25 --top-k 10
#
# Referencia BM25/RAG: Lewis et al. 2020 (DOI: 10.48550/arXiv.2005.11401)
# ─────────────────────────────────────────────────────────────────────────────


# ─── Helpers de prompt ────────────────────────────────────────────────────────

def _extract_abstract(text: str, fallback_words: int = 400) -> str:
    """
    Extrae la sección Abstract real del paper buscando el encabezado 'Abstract'
    y cortando en el siguiente encabezado de sección (Introduction, Methods, etc.).
    Fallback: primeras fallback_words palabras si no se encuentra.
    """
    # Encabezados que marcan el fin del abstract
    END_SECTIONS = re.compile(
        r'^\s*(introduction|background|methods?|materials?\s+and\s+methods?|'
        r'results?|discussion|keywords?|1[\.\s]|2[\.\s])',
        re.IGNORECASE | re.MULTILINE,
    )
    # Buscar inicio del abstract
    start_m = re.search(r'(?:^|\n)\s*abstract\s*\n', text, re.IGNORECASE)
    if start_m:
        body = text[start_m.end():]
        end_m = END_SECTIONS.search(body)
        snippet = body[:end_m.start()].strip() if end_m else body[:3000].strip()
        if snippet:
            return snippet
    # Fallback: primeras N palabras
    return " ".join(text.split()[:fallback_words])


def _fmt_abstract(text: str, max_words: int = 0) -> str:
    """
    Formatea el abstract real del paper para inyectarlo en prompts.
    max_words ignorado — se usa detección de sección.
    """
    if not text or not text.strip():
        return ""
    snippet = _extract_abstract(text)
    return f"PAPER ABSTRACT:\n{snippet}\n\n"


def _fmt_full_context(text: str, max_words: int = 0) -> str:
    """
    Formatea el texto completo del paper para el modo anchored=full.
    max_words=0 → sin límite.
    """
    if not text or not text.strip():
        return ""
    words = text.split()
    total = len(words)
    if max_words > 0 and total > max_words:
        words = words[:max_words]
        truncated = True
    else:
        truncated = False
    snippet = " ".join(words)
    note = f" [truncated at {max_words} words, original: {total}]" if truncated else f" [{len(words)} words]"
    return snippet, note


def _parse_json(text: str) -> dict | None:
    """Extrae JSON válido de la respuesta del LLM (maneja markdown fences)."""
    clean = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except Exception:
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


# ─── Prompts ──────────────────────────────────────────────────────────────────

PROMPT_INFERENCE_FIG_TEMPLATE = """{abstract_block}Figure caption: {caption}

Analyze this scientific figure. Respond ONLY with valid JSON — no markdown fences, no text outside the JSON object:

{{
  "figure_type": "bar chart | scatter plot | line graph | Western blot | microscopy | heatmap | survival curve | flow cytometry | schematic | other",
  "visual_description": "panels, axes, labels, colors, units, legends, organisms/structures shown. 2-3 sentences.",
  "groups_compared": "conditions, treatments, timepoints, genotypes or cell lines contrasted in the figure.",
  "statistical_markers": "every visible statistical element: n=, error bars type (SD/SEM/95%CI), p-values, R², fold-changes. Write 'None visible' if absent.",
  "key_finding": "the main result shown, citing specific numbers from the image when visible.",
  "caption_accurate": true,
  "caption_discrepancy": "elements in image not described by caption, or caption claims not supported visually. Write 'None' if accurate.",
  "scientific_interpretation": "what biological or scientific question this figure addresses and what the data demonstrates. Be specific about mechanism, pathway, or phenomenon.",
  "confidence": "high | medium | low"
}}

confidence: high=evidence clearly readable in image; medium=partially visible; low=inferring beyond what is shown.
If multi-panel (A, B, C...), address all panels in visual_description and key_finding.
Never refuse — all figures must be analyzed."""

PROMPT_INFERENCE_TBL_TEMPLATE = """{abstract_block}Table caption: {caption}

Analyze this scientific table. Respond ONLY with valid JSON — no markdown fences, no text outside the JSON object:

{{
  "table_type": "results comparison | parameter table | patient demographics | statistical summary | ablation study | other",
  "structure": "columns and rows described: what is compared, against what baselines, using what metric and units.",
  "statistical_markers": "significance markers (*, **, ***), p-values, CIs, n=, standard deviations visible. Write 'None visible' if absent.",
  "best_result": "the row or cell showing the strongest or most notable result, with its exact value.",
  "key_pattern": "the main trend, comparison or contrast that stands out across the table.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between caption and actual table content. Write 'None' if accurate.",
  "scientific_interpretation": "what question this table addresses and what the numbers prove. Cite specific values.",
  "confidence": "high | medium | low"
}}

confidence: high=all values clearly readable; medium=some cells hard to read; low=inferring.
Never refuse — all tables must be analyzed."""


# Anchored — texto completo del paper como contexto
PROMPT_ANCHORED_FULL_FIG_TEMPLATE = """{abstract_block}FULL PAPER TEXT{context_note}:
{context}

---

Figure caption: {caption}

Using the full paper text above as authoritative reference, analyze this figure. Respond ONLY with valid JSON:

{{
  "figure_type": "bar chart | scatter | Western blot | microscopy | heatmap | survival curve | other",
  "visual_description": "panels, axes, labels, colors, units, organisms visible. 2-3 sentences.",
  "hypothesis_tested": "the specific claim or hypothesis from the paper this figure tests. Quote the relevant sentence.",
  "paper_quote": "exact sentence from the paper text that this figure is meant to support.",
  "key_findings": "specific quantitative results visible in the figure, connected to numbers in the text.",
  "statistical_markers": "every visible statistical element: n=, error bars (SD/SEM/CI), p-values, R², significance markers. Write 'None visible' if absent.",
  "controls_assessment": "experimental controls present (positive, negative, baseline). Note absent controls given the experimental design.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between caption and visual. Write 'None' if accurate.",
  "scientific_conclusion": "what this figure definitively demonstrates, what alternative explanations it rules out, and its role in the paper's argument.",
  "confidence": "high | medium | low"
}}

confidence: high=clear visual evidence + strong paper alignment; medium=partial; low=inferring.
Never refuse."""

PROMPT_ANCHORED_FULL_TBL_TEMPLATE = """{abstract_block}FULL PAPER TEXT{context_note}:
{context}

---

Table caption: {caption}

Using the full paper text above as authoritative reference, analyze this table. Respond ONLY with valid JSON:

{{
  "table_type": "results comparison | parameter table | patient demographics | statistical summary | ablation | other",
  "structure": "what is compared, by what metric, against what baselines.",
  "paper_quote": "exact sentence from the paper text that this table is meant to support.",
  "key_entries": "most relevant rows and values given the paper's claims. Cite specific numbers.",
  "statistical_markers": "significance markers (*, **, ***), p-values, CIs, n= visible. Write 'None visible' if absent.",
  "controls_assessment": "baseline or reference conditions used. Note missing controls given the context.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between caption and actual table content. Write 'None' if accurate.",
  "scientific_conclusion": "what this table definitively demonstrates, what it rules out, its role in the paper's argument.",
  "confidence": "high | medium | low"
}}

confidence: high=values clearly readable + strong paper alignment; medium=partial; low=inferring.
Never refuse."""

# Anchored — BM25 top-k chunks (modo bm25)
PROMPT_ANCHORED_BM25_FIG_TEMPLATE = """{abstract_block}RELEVANT PAPER SECTIONS (BM25-retrieved, top-{top_k} chunks for this figure):
{context}

---

Figure caption: {caption}

Analyze this figure using the retrieved paper sections as reference. Respond ONLY with valid JSON:

{{
  "figure_type": "bar chart | scatter | Western blot | microscopy | heatmap | survival curve | other",
  "visual_description": "panels, axes, labels, colors, units, organisms visible. 2-3 sentences.",
  "hypothesis_tested": "the specific claim or hypothesis from the paper this figure tests. Quote the relevant sentence.",
  "paper_quote": "exact sentence from the retrieved text that this figure is meant to support.",
  "key_findings": "specific quantitative results visible in the figure, connected to numbers in the retrieved text.",
  "statistical_markers": "every visible statistical element: n=, error bars (SD/SEM/CI), p-values, R², significance markers. Write 'None visible' if absent.",
  "controls_assessment": "experimental controls present. Note absent controls given the retrieved context.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between caption and visual. Write 'None' if accurate.",
  "scientific_conclusion": "what this figure definitively demonstrates, what alternative explanations it rules out, and its role in the paper's argument.",
  "confidence": "high | medium | low"
}}

Never refuse."""

PROMPT_ANCHORED_BM25_TBL_TEMPLATE = """{abstract_block}RELEVANT PAPER SECTIONS (BM25-retrieved, top-{top_k} chunks for this table):
{context}

---

Table caption: {caption}

Analyze this table using the retrieved paper sections as reference. Respond ONLY with valid JSON:

{{
  "table_type": "results comparison | parameter table | patient demographics | statistical summary | ablation | other",
  "structure": "what is compared, by what metric, against what baselines.",
  "paper_quote": "exact sentence from the retrieved text that this table is meant to support.",
  "key_entries": "most relevant rows and values given the paper's claims. Cite specific numbers.",
  "statistical_markers": "significance markers (*, **, ***), p-values, CIs, n= visible. Write 'None visible' if absent.",
  "controls_assessment": "baseline or reference conditions used. Note missing controls.",
  "caption_accurate": true,
  "caption_discrepancy": "discrepancy between caption and actual table content. Write 'None' if accurate.",
  "scientific_conclusion": "what this table definitively demonstrates, what it rules out, its role in the paper's argument.",
  "confidence": "high | medium | low"
}}

Never refuse."""


# ─── HTTP client ──────────────────────────────────────────────────────────────
def ask_api(server, prompt, image_bytes=None, max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE, timeout=DEFAULT_TIMEOUT):
    content = []
    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "messages":    [{"role": "user", "content": content}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      False,
    }

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(server, json=payload, timeout=timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            if not text:
                raise ValueError("empty response")
            return text
        except Exception as e:
            last_err = e
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"    retry {attempt + 1}/{MAX_RETRIES}: {e} ({wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed after {MAX_RETRIES}: {last_err}")


def server_health(server):
    try:
        url = server.rsplit("/v1/", 1)[0] + "/health"
        return requests.get(url, timeout=5).status_code == 200
    except Exception:
        return False


# ─── Text extraction ──────────────────────────────────────────────────────────
def extract_paper_text(pdf_path=None, context_file=None):
    """Extrae texto del PDF o lo lee desde un archivo .txt pre-extraído."""
    if context_file:
        return Path(context_file).read_text(encoding="utf-8")
    doc = fitz.open(str(pdf_path))
    parts = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            parts.append(f"[Page {i + 1}]\n{text}")
    doc.close()
    return "\n\n".join(parts)


# ─── RAG: chunking (solo para context_strategy="bm25") ───────────────────────
def _tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def chunk_text(text, chunk_words=DEFAULT_CHUNK_WORDS, overlap=DEFAULT_CHUNK_OVERLAP):
    """Divide el texto en chunks con overlap. Usado solo en modo bm25."""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current_words = []
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        if len(words) > chunk_words * 1.5:
            for start in range(0, len(words), chunk_words - overlap):
                segment = words[start:start + chunk_words]
                if len(segment) >= 20:
                    chunks.append(" ".join(segment))
            continue
        if len(current_words) + len(words) > chunk_words:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] + words
        else:
            current_words.extend(words)
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def _build_bm25(tokenized_chunks):
    try:
        from rank_bm25 import BM25Okapi
        index = BM25Okapi(tokenized_chunks)
        return lambda q: index.get_scores(q)
    except ImportError:
        n = len(tokenized_chunks)
        df = {}
        for tokens in tokenized_chunks:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        def score_fn(query_tokens):
            scores = []
            for tokens in tokenized_chunks:
                tf_map = {}
                for t in tokens:
                    tf_map[t] = tf_map.get(t, 0) + 1
                s = 0.0
                for qt in query_tokens:
                    if qt in tf_map:
                        tf  = tf_map[qt] / max(len(tokens), 1)
                        idf = math.log((n + 1) / (df.get(qt, 0) + 1)) + 1
                        s  += tf * idf
                scores.append(s)
            return scores

        return score_fn


def build_index(chunks):
    tokenized = [_tokenize(c) for c in chunks]
    return _build_bm25(tokenized), tokenized


def retrieve(query, score_fn, chunks, top_k=DEFAULT_TOP_K):
    query_tokens = _tokenize(query)
    if not query_tokens:
        return chunks[:top_k]
    scores     = score_fn(query_tokens)
    ranked     = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_idx    = sorted([i for i, _ in ranked[:top_k]])
    return [chunks[i] for i in top_idx]


def build_rag_index(text, chunk_words=DEFAULT_CHUNK_WORDS, overlap=DEFAULT_CHUNK_OVERLAP):
    chunks = chunk_text(text, chunk_words=chunk_words, overlap=overlap)
    score_fn, _ = build_index(chunks)
    return chunks, score_fn


# ─── Pipeline ─────────────────────────────────────────────────────────────────
def analyze_all(
    figures_json,
    pdf_path=None,
    context_file=None,
    server=DEFAULT_SERVER,
    mode_inference=True,
    mode_anchored=True,
    context_strategy=DEFAULT_CONTEXT_STRATEGY,   # "full" | "bm25"
    max_context_words=DEFAULT_MAX_CONTEXT_WORDS,  # 0 = sin límite (solo en full)
    abstract_words=DEFAULT_ABSTRACT_WORDS,        # 0 = abstract completo
    chunk_words=DEFAULT_CHUNK_WORDS,              # solo en bm25
    overlap=DEFAULT_CHUNK_OVERLAP,                # solo en bm25
    top_k=DEFAULT_TOP_K,                          # solo en bm25
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    timeout=DEFAULT_TIMEOUT,
    out_path=None,
    tables_json=None,
):
    fig_json = Path(figures_json)
    if pdf_path:
        pdf_path = Path(pdf_path)

    meta  = json.loads(fig_json.read_text(encoding="utf-8"))
    items = list(meta["items"])

    # Merge tablas de extract_tables.py
    if tables_json:
        tbl_path = Path(tables_json)
        if tbl_path.exists():
            tbl_meta = json.loads(tbl_path.read_text(encoding="utf-8"))
            items.extend(tbl_meta["items"])

    if out_path is None:
        out_path = fig_json.parent / "analyses_rag.json"
    else:
        out_path = Path(out_path)

    # Resume: saltar ítems ya analizados
    results = []
    done    = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for it in prev.get("items", []):
                has_inf  = "inference" in it
                has_anch = "anchored"  in it
                ok = (not mode_inference or has_inf) and (not mode_anchored or has_anch)
                if ok:
                    results.append(it)
                    done.add(it["label"])
        except Exception:
            pass

    n_figs = sum(1 for it in items if it.get("kind") != "table")
    n_tbls = sum(1 for it in items if it.get("kind") == "table")
    print(f"\nServer: {server}")
    print(f"Items: {len(items)} ({n_figs} figuras + {n_tbls} tablas) | ya hechos: {len(done)}")
    print(f"Modos: inference={mode_inference} anchored={mode_anchored}")
    print(f"Estrategia: context_strategy={context_strategy} | abstract_words={abstract_words or 'completo'}")
    if context_strategy == "bm25":
        print(f"BM25: top_k={top_k} | chunk_words={chunk_words} | overlap={overlap}")
    if context_strategy == "full" and max_context_words:
        print(f"Full-text: max_context_words={max_context_words}")
    print()

    # ── Extraer texto del paper (una sola vez) ──────────────────────────────
    raw_text = ""
    try:
        if context_file and Path(context_file).exists():
            raw_text = Path(context_file).read_text(encoding="utf-8", errors="ignore")
        elif pdf_path:
            raw_text = extract_paper_text(pdf_path=pdf_path)
    except Exception as e:
        print(f"  WARN texto: {e}")

    # ── Abstract block (inyectado en prompts de inferencia pura) ───────────
    abstract_block = _fmt_abstract(raw_text, max_words=abstract_words)
    if abstract_block:
        wc = len(abstract_block.split())
        print(f"  Abstract block: {wc} palabras → inyectado en todos los prompts")

    # ── Preparar contexto para modo anchored ────────────────────────────────
    # full: texto completo (truncado si max_context_words > 0)
    # bm25: construir índice; el retrieve se hace por ítem
    full_context_text   = None
    full_context_note   = ""
    chunks              = None
    score_fn            = None

    if mode_anchored and raw_text:
        if context_strategy == "full":
            full_context_text, full_context_note = _fmt_full_context(
                raw_text, max_words=max_context_words
            )
            wc = len(full_context_text.split())
            print(f"  Full-text context: {wc} palabras{full_context_note}")
        else:  # bm25
            print(f"  Construyendo índice BM25 sobre {len(raw_text):,} chars...")
            chunks = chunk_text(raw_text, chunk_words=chunk_words, overlap=overlap)
            score_fn, _ = build_index(chunks)
            print(f"  Chunks: {len(chunks)} ({chunk_words}w c/u, {overlap}w overlap)")
    print()

    # ── Loop principal ──────────────────────────────────────────────────────
    for i, item in enumerate(items):
        if item["label"] in done:
            print(f"[{i + 1}/{len(items)}] {item['label']} SKIP")
            continue

        img_path = Path(item["image_path"])
        if not img_path.is_absolute():
            img_path = fig_json.parent / img_path.name
        img_bytes = img_path.read_bytes()

        is_table = item.get("kind") == "table"
        caption  = item.get("caption", "Not provided.")
        result   = dict(item)
        t_start  = time.time()

        # 1) Inferencia pura (imagen + caption + abstract)
        if mode_inference:
            tmpl_inf   = PROMPT_INFERENCE_TBL_TEMPLATE if is_table else PROMPT_INFERENCE_FIG_TEMPLATE
            prompt_inf = tmpl_inf.format(caption=caption, abstract_block=abstract_block)
            try:
                ans = ask_api(server, prompt_inf, image_bytes=img_bytes,
                              max_tokens=max_tokens, temperature=temperature, timeout=timeout)
                result["inference"] = ans
                parsed = _parse_json(ans)
                if parsed:
                    result["inference_parsed"] = parsed
                    result["confidence"]       = parsed.get("confidence", "unknown")
            except Exception as e:
                result["inference_error"] = str(e)

        # 2) Anchored (imagen + caption + texto completo o chunks BM25)
        if mode_anchored and raw_text:
            if context_strategy == "full":
                tmpl = (PROMPT_ANCHORED_FULL_TBL_TEMPLATE if is_table
                        else PROMPT_ANCHORED_FULL_FIG_TEMPLATE)
                prompt_anc = tmpl.format(
                    context=full_context_text,
                    context_note=full_context_note,
                    caption=caption,
                    abstract_block=abstract_block,
                )
                result["context_words"]    = len(full_context_text.split())
                result["context_strategy"] = "full"
            else:  # bm25
                retrieved  = retrieve(caption, score_fn, chunks, top_k=top_k)
                context    = "\n\n---\n\n".join(retrieved)
                tmpl = (PROMPT_ANCHORED_BM25_TBL_TEMPLATE if is_table
                        else PROMPT_ANCHORED_BM25_FIG_TEMPLATE)
                prompt_anc = tmpl.format(
                    context=context,
                    caption=caption,
                    abstract_block=abstract_block,
                    top_k=top_k,
                )
                result["retrieved_chunks"] = len(retrieved)
                result["retrieved_words"]  = sum(len(c.split()) for c in retrieved)
                result["context_strategy"] = "bm25"

            try:
                ans = ask_api(server, prompt_anc, image_bytes=img_bytes,
                              max_tokens=max_tokens, temperature=temperature, timeout=timeout)
                result["anchored"] = ans
                parsed = _parse_json(ans)
                if parsed:
                    result["anchored_parsed"] = parsed
                    result["confidence"]      = parsed.get("confidence",
                                                           result.get("confidence", "unknown"))
            except Exception as e:
                result["anchored_error"] = str(e)

        result["elapsed_sec"] = round(time.time() - t_start, 1)
        results.append(result)

        ctx_info = ""
        if context_strategy == "full":
            ctx_info = f" [{result.get('context_words', '?')}w full]"
        elif context_strategy == "bm25":
            ctx_info = f" [{result.get('retrieved_chunks', '?')} chunks]"
        preview = (result.get("inference") or result.get("anchored") or "")[:80].replace("\n", " ")
        kind_tag = "TBL" if is_table else "FIG"
        print(f"[{i + 1}/{len(items)}] [{kind_tag}] {item['label']} p{item['page']} "
              f"({result['elapsed_sec']}s){ctx_info} {preview}...")

        out_path.write_text(
            json.dumps({
                "total":            len(results),
                "context_strategy": context_strategy,
                "abstract_words":   abstract_words or "full",
                "max_context_words": max_context_words or "unlimited",
                "top_k":            top_k if context_strategy == "bm25" else None,
                "chunk_words":      chunk_words if context_strategy == "bm25" else None,
                "items":            results,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nResultados guardados: {out_path}")
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Analiza figuras Y tablas científicas con LLM multimodal + contexto RAG/full.")
    p.add_argument("figures_json",      help="figures.json de extract_figures.py")
    p.add_argument("--pdf",             help="PDF original")
    p.add_argument("--context-file",    help="Texto plano pre-extraído del paper (.txt)")
    p.add_argument("--server",          default=DEFAULT_SERVER)
    p.add_argument("--inference-only",  action="store_true")
    p.add_argument("--anchored-only",   action="store_true")
    p.add_argument("--out",             help="ruta de salida JSON")

    # Estrategia de contexto
    p.add_argument("--context-strategy", choices=["full", "bm25"],
                   default=DEFAULT_CONTEXT_STRATEGY,
                   help=f"Estrategia anchored: 'full'=texto completo (def) | 'bm25'=top-k chunks")
    p.add_argument("--max-context-words", type=int, default=DEFAULT_MAX_CONTEXT_WORDS,
                   help="Palabras máx del texto completo en modo full (0=sin límite)")
    p.add_argument("--abstract-words",  type=int, default=DEFAULT_ABSTRACT_WORDS,
                   help="Palabras del abstract en prompts de inferencia (0=completo)")

    # Parámetros BM25 (solo si --context-strategy bm25)
    p.add_argument("--top-k",           type=int, default=DEFAULT_TOP_K,
                   help=f"Chunks BM25 por ítem (solo bm25, def: {DEFAULT_TOP_K})")
    p.add_argument("--chunk-words",     type=int, default=DEFAULT_CHUNK_WORDS,
                   help=f"Palabras por chunk BM25 (def: {DEFAULT_CHUNK_WORDS})")
    p.add_argument("--chunk-overlap",   type=int, default=DEFAULT_CHUNK_OVERLAP,
                   help=f"Overlap entre chunks BM25 (def: {DEFAULT_CHUNK_OVERLAP})")

    # LLM
    p.add_argument("--max-tokens",      type=int,   default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature",     type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--timeout",         type=int,   default=DEFAULT_TIMEOUT)
    args = p.parse_args(argv)

    if not server_health(args.server):
        sys.exit(f"Servidor no responde: {args.server}")

    mode_inf = not args.anchored_only
    mode_anc = not args.inference_only

    if mode_anc and not args.pdf and not args.context_file:
        sys.exit("Requiere --pdf o --context-file para anchored. Usa --inference-only si no tenés el paper.")

    analyze_all(
        args.figures_json,
        pdf_path=args.pdf,
        context_file=args.context_file,
        server=args.server,
        mode_inference=mode_inf,
        mode_anchored=mode_anc,
        context_strategy=args.context_strategy,
        max_context_words=args.max_context_words,
        abstract_words=args.abstract_words,
        chunk_words=args.chunk_words,
        overlap=args.chunk_overlap,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
