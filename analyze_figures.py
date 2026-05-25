"""
LLM Analyzer for Extracted Figures
====================================

Toma el output de extract_figures.py y analiza cada figura/tabla con un
modelo visión-lenguaje vía API OpenAI-compatible.

Genera DOS análisis por item:
  1. inference   — solo imagen + prompt estructurado (inferencia pura)
  2. anchored    — imagen + contexto del paper (anclado al paper)

CONTEXTO INTELIGENTE:
  - Si el paper cabe en (context_window − budget) → usa paper completo
  - Si no cabe → el LLM resume el paper UNA VEZ y reutiliza ese resumen

Salida orientada a TRAINING DATA: prompts estructurados que fuerzan
análisis multi-aspecto, evitan respuestas evasivas.

Uso:
    python analyze_figures.py extracted/figures.json --pdf paper.pdf \\
        --server http://127.0.0.1:8080/v1/chat/completions

    python analyze_figures.py extracted/figures.json --inference-only
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests


# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_SERVER         = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MAX_TOKENS     = 1200
DEFAULT_TEMPERATURE    = 0.2
DEFAULT_TIMEOUT        = 300
DEFAULT_CONTEXT_BUDGET = 6000   # tokens reservados para texto del paper
MAX_RETRIES            = 5
RETRY_BACKOFF          = 3


# ─── Prompts orientados a training data ─────────────────────────────────────

# Inferencia pura: solo la imagen, sin info del paper
PROMPT_INFERENCE_FIG = """You are creating high-quality training labels for a vision-language model that analyzes scientific figures. Your output will be used as ground truth — be precise, thorough, and never refuse.

Analyze this scientific figure following EXACTLY this structure (every section is required):

## Visual Description
What is visible: panels, axes, colors, labels, legends, units, symbols, organisms/structures shown. 2-4 sentences.

## Figure Type
Type of visualization (bar chart, scatter plot, schematic, microscopy, gel, heatmap, workflow diagram, etc.) and what experimental data it represents.

## Data and Patterns
Specific values, trends, comparisons, or relationships shown. Cite numbers visible in the image when possible. Identify groups being compared.

## Scientific Interpretation
What biological/scientific question this figure addresses and what conclusion the data supports. Be specific about mechanism, pathway, or phenomenon.

## Significance
Why this finding matters. What it proves, suggests, or rules out.

If multi-panel (A, B, C, ...), briefly address each panel's contribution. Never refuse — even minimal figures must be analyzed."""

PROMPT_INFERENCE_TBL = """You are creating training labels for a vision-language model. Your output is ground truth — be precise, thorough, never refuse.

Analyze this scientific table following EXACTLY this structure:

## Table Description
Columns, rows, what is being compared. Units and scale.

## Data Summary
Most important entries — best/worst values, surprising results, notable patterns.

## Scientific Interpretation
What this table proves or argues. What biological/methodological insight emerges from the comparison.

## Significance
Why this comparison matters in the paper's context."""


# Anclado al paper: imagen + paper como contexto
PROMPT_ANCHORED_FIG_TEMPLATE = """You are creating training labels using both a scientific figure and the paper it comes from. Your output is ground truth — be precise, never refuse.

PAPER TEXT:
{context}

---

Analyze the figure below, USING THE PAPER as authoritative reference. Follow EXACTLY this structure:

## Visual Description
What is visible in the figure (panels, axes, symbols, organisms).

## Hypothesis Tested
The specific claim or hypothesis from the paper that this figure tests or supports. Quote relevant text.

## Key Data and Findings
Specific quantitative results (values, fold-changes, p-values, gene names) that this figure demonstrates. Connect visual elements to numbers in the paper text.

## Mechanistic Insight
What the data reveals about the biology, pathway, or system being studied. Reference the paper's argument.

## Narrative Role
How this figure advances the paper's overall argument (establishes baseline, demonstrates causation, validates model, etc.).

## Limitations / Caveats
Any limitations or alternative interpretations visible in the data or noted in the paper.

## Scientific Conclusion
Synthesize the visual evidence with the paper's framework into a precise, self-contained scientific conclusion. State what this figure definitively demonstrates, what alternative explanations it rules out, and its conceptual significance in the paper's broader argument. Write as a scientist who has fully internalized both the data and the theory — be rigorous, specific, and let the conclusion feel earned by the evidence."""

PROMPT_ANCHORED_TBL_TEMPLATE = """You are creating training labels using a scientific table and its paper. Be precise, never refuse.

PAPER TEXT:
{context}

---

Analyze the table below using the paper as reference:

## Table Description
What is compared, by what metric, against what baselines.

## Key Entries
Most relevant rows/columns given the paper's claims. Cite specific values.

## Argument Supported
What conclusion from the paper this table substantiates. Quote relevant text.

## Significance
Why this comparison matters in the paper's narrative.

## Scientific Conclusion
Synthesize the table's data with the paper's framework into a precise scientific conclusion. State what this table definitively demonstrates, what it rules out, and its role in the paper's argument. Be rigorous and specific — let the conclusion be earned by the numbers."""


# Para resumir paper largo (solo texto, sin imagen)
PROMPT_SUMMARIZE = """Summarize this scientific paper for downstream figure analysis. Include:

## Research Question
The main scientific question and hypothesis.

## Methods Overview
Key experimental approaches and model systems used.

## Main Findings
The primary results, with specific values, gene names, mechanisms when mentioned.

## Per-Figure Context
For each figure/table mentioned in the text, briefly describe what it shows and its role.

## Conclusions and Significance
What the paper concludes and why it matters.

Be thorough but concise (target 800-1000 words). Preserve technical specificity.

PAPER:
{paper_text}"""


# ─── HTTP client ─────────────────────────────────────────────────────────────
def ask_api(server, prompt, image_bytes=None, max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE, timeout=DEFAULT_TIMEOUT):
    """Generic chat completion. With or without image."""
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


# ─── Paper context: smart full-vs-summary ────────────────────────────────────
def extract_full_paper_text(pdf_path):
    doc = fitz.open(str(pdf_path))
    parts = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            parts.append(f"[Page {i + 1}]\n{text}")
    doc.close()
    return "\n\n".join(parts)


def estimate_tokens(text):
    """Aproximación conservadora: 1 token ≈ 3.5 chars en inglés."""
    return len(text) // 3


def compute_paper_context(pdf_path, server, budget_tokens=DEFAULT_CONTEXT_BUDGET):
    """
    Retorna (context_text, mode) donde mode ∈ {"full", "summary"}.
    Si el paper entero cabe en budget_tokens → "full".
    Si no → llama al LLM para resumirlo y retorna "summary".
    """
    full_text = extract_full_paper_text(pdf_path)
    est = estimate_tokens(full_text)

    print(f"Paper: {len(full_text):,} chars (~{est:,} tokens) | budget: {budget_tokens:,} tokens")

    if est <= budget_tokens:
        print("  Cabe entero -> usando paper completo como contexto")
        return full_text, "full"

    print("  Demasiado grande -> generando resumen via LLM (una sola vez)...")
    # Truncar input para que quepa en una request de resumen
    truncated = full_text[:budget_tokens * 3]  # ~chars de budget tokens
    summary = ask_api(server, PROMPT_SUMMARIZE.format(paper_text=truncated),
                      max_tokens=1500, temperature=0.3)
    print(f"  Resumen: {len(summary):,} chars (~{estimate_tokens(summary):,} tokens)")
    return summary, "summary"


# ─── Pipeline ────────────────────────────────────────────────────────────────
def analyze_all(figures_json, pdf_path, server, mode_inference=True, mode_anchored=True,
                budget_tokens=DEFAULT_CONTEXT_BUDGET, max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE, timeout=DEFAULT_TIMEOUT, out_path=None):
    fig_json = Path(figures_json)
    pdf_path = Path(pdf_path) if pdf_path else None
    meta = json.loads(fig_json.read_text(encoding="utf-8"))
    items = meta["items"]

    if out_path is None:
        out_path = fig_json.parent / "analyses.json"
    else:
        out_path = Path(out_path)

    # Resume
    results = []
    done = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for it in prev.get("items", []):
                has_inference = "inference" in it
                has_anchored  = "anchored"  in it
                # Solo skip si tiene todos los modos solicitados
                ok = (not mode_inference or has_inference) and (not mode_anchored or has_anchored)
                if ok:
                    results.append(it)
                    done.add(it["label"])
        except Exception:
            pass

    print(f"\nServer: {server}")
    print(f"Items: {len(items)} | hechos: {len(done)}")
    print(f"Modos: inference={mode_inference} anchored={mode_anchored}\n")

    # Pre-computar contexto del paper (una sola vez)
    paper_context = None
    context_mode  = None
    if mode_anchored and pdf_path and pdf_path.exists():
        paper_context, context_mode = compute_paper_context(pdf_path, server, budget_tokens)
        print()

    for i, item in enumerate(items):
        if item["label"] in done:
            print(f"[{i + 1}/{len(items)}] {item['label']} SKIP")
            continue

        img_path = Path(item["image_path"])
        if not img_path.is_absolute():
            img_path = fig_json.parent / img_path.name
        img_bytes = img_path.read_bytes()

        is_table = item["kind"] == "table"
        result = dict(item)
        t_start = time.time()

        # 1) Inferencia pura (solo imagen)
        if mode_inference:
            prompt_inf = PROMPT_INFERENCE_TBL if is_table else PROMPT_INFERENCE_FIG
            try:
                ans = ask_api(server, prompt_inf, image_bytes=img_bytes,
                              max_tokens=max_tokens, temperature=temperature, timeout=timeout)
                result["inference"] = ans
            except Exception as e:
                result["inference_error"] = str(e)

        # 2) Anclado al paper (imagen + contexto)
        if mode_anchored and paper_context:
            tmpl = PROMPT_ANCHORED_TBL_TEMPLATE if is_table else PROMPT_ANCHORED_FIG_TEMPLATE
            prompt_anc = tmpl.format(context=paper_context)
            try:
                ans = ask_api(server, prompt_anc, image_bytes=img_bytes,
                              max_tokens=max_tokens, temperature=temperature, timeout=timeout)
                result["anchored"] = ans
            except Exception as e:
                result["anchored_error"] = str(e)

        result["elapsed_sec"]  = round(time.time() - t_start, 1)
        result["context_mode"] = context_mode
        results.append(result)

        preview = (result.get("inference") or result.get("anchored") or "")[:90].replace("\n", " ")
        print(f"[{i + 1}/{len(items)}] {item['label']} p{item['page']} ({result['elapsed_sec']}s) {preview}...")

        # Guardado incremental
        out_path.write_text(
            json.dumps({
                "total":        len(results),
                "context_mode": context_mode,
                "items":        results,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nResultados: {out_path}")
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description="Analiza figuras con LLM via API OpenAI-compatible.")
    p.add_argument("figures_json", help="figures.json producido por extract_figures.py")
    p.add_argument("--pdf",        help="PDF original (para análisis anchored)")
    p.add_argument("--server",     default=DEFAULT_SERVER)
    p.add_argument("--inference-only", action="store_true",
                   help="solo inferencia pura, sin contexto del paper")
    p.add_argument("--anchored-only",  action="store_true",
                   help="solo anclado al paper, requiere --pdf")
    p.add_argument("--out",         help="ruta de salida (def: analyses.json)")
    p.add_argument("--budget-tokens", type=int, default=DEFAULT_CONTEXT_BUDGET,
                   help=f"tokens reservados para texto del paper (def: {DEFAULT_CONTEXT_BUDGET})")
    p.add_argument("--max-tokens",  type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--timeout",     type=int, default=DEFAULT_TIMEOUT)
    args = p.parse_args(argv)

    if not server_health(args.server):
        sys.exit(f"Servidor no responde: {args.server}")

    mode_inf = not args.anchored_only
    mode_anc = not args.inference_only

    if mode_anc and not args.pdf:
        sys.exit("--pdf requerido para anchored. Usa --inference-only si no tienes el PDF.")

    analyze_all(
        args.figures_json,
        pdf_path=args.pdf,
        server=args.server,
        mode_inference=mode_inf,
        mode_anchored=mode_anc,
        budget_tokens=args.budget_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
