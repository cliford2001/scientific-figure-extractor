"""
Pipeline completo: DB (parquet) → PDF → figuras → análisis RAG
===============================================================

Para cada paper en el parquet:
  1. Lee doi, pmcid y text_clean del DB
  2. Descarga el PDF desde PubMed Central
  3. Extrae figuras con extract_figures.py
  3b. Extrae tablas con extract_tables.py (find_tables)
  4. Usa text_clean del DB como contexto (sin re-extraer del PDF)
  5. Analiza con analyze_figures_v2_rag.py (inference + anchored RAG)
  6. Elimina el PDF
  7. Conserva: figuras extraídas + analysis.json

Salida por paper:
  out-dir/
  └── {pmcid}/
      ├── p002_Figure_1.png
      ├── p007_Table_1.png
      ├── figures.json
      ├── tables.json
      └── analysis.json

Uso:
    python pipeline_db_to_analysis.py \\
        --metadata ibio-...-metadata-part_0001.parquet \\
        --texts    ibio-...-texts-part_0001.parquet \\
        --out-dir  resultados/ \\
        --server   http://146.155.169.90:8080/v1/chat/completions \\
        --limit    10

    # Filtrar por campo y calidad mínima
    python pipeline_db_to_analysis.py \\
        --metadata meta.parquet --texts texts.parquet \\
        --field Agriculture --quality-min 3 --limit 50
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import requests

# ─── Importar los dos scripts del pipeline ───────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import extract_figures as ef
import extract_tables as et
import analyze_figures_v2_rag as rag


# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SERVER   = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_OUT_DIR  = "resultados"
DEFAULT_DELAY    = 2.0     # segundos entre descargas (respetar rate limit PMC)
DEFAULT_TOP_K    = 5
DEFAULT_CHUNK_W  = 250
DEFAULT_OVERLAP  = 40
DEFAULT_QUALITY  = 0       # quality_score mínimo (0 = todos)


# ─── Descarga PDF via Unpaywall (DOI → PDF URL) ──────────────────────────────
UNPAYWALL_EMAIL = "fernver62@gmail.com"

def _get_pdf_url_unpaywall(doi: str, timeout: int = 15) -> str | None:
    """Consulta Unpaywall para obtener la URL del PDF open access."""
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    url = f"https://api.unpaywall.org/v2/{doi_clean}?email={UNPAYWALL_EMAIL}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        loc  = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        return None


def download_pdf(pmcid: str, doi: str, out_path: Path, timeout: int = 60) -> bool:
    headers = {"User-Agent": "scientific-figure-pipeline/1.0 (research use)"}

    # 1) Unpaywall via DOI
    if doi:
        pdf_url = _get_pdf_url_unpaywall(doi)
        if pdf_url:
            try:
                r = requests.get(pdf_url, headers=headers, allow_redirects=True,
                                 timeout=timeout)
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    out_path.write_bytes(r.content)
                    return True
                print(f"    Unpaywall URL no devolvió PDF ({r.status_code}): {pdf_url[:70]}")
            except Exception as e:
                print(f"    Unpaywall descarga fallida: {e}")

    # 2) Fallback: PMC directo via pmcid
    pmcid_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    try:
        r = requests.get(pmcid_url, headers=headers, allow_redirects=True, timeout=timeout)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            out_path.write_bytes(r.content)
            return True
    except Exception as e:
        print(f"    PMC fallback fallido: {e}")

    return False


# ─── Carga y filtrado del parquet ─────────────────────────────────────────────
def load_papers(metadata_path: str, texts_path: str,
                field: str | None, quality_min: int,
                limit: int, offset: int):
    try:
        import pandas as pd
    except ImportError:
        sys.exit("Requiere pandas: pip install pandas pyarrow")

    meta  = pd.read_parquet(metadata_path)
    texts = pd.read_parquet(texts_path, columns=["pmcid", "text_clean"])
    df    = meta.merge(texts, on="pmcid")

    if field:
        df = df[df["main_field"].str.lower() == field.lower()]
    if quality_min > 0:
        df = df[df["quality_score"] >= quality_min]

    df = df.iloc[offset:offset + limit] if limit else df.iloc[offset:]

    print(f"Papers a procesar: {len(df):,}"
          + (f" | field={field}" if field else "")
          + (f" | quality≥{quality_min}" if quality_min else ""))
    return df


# ─── Pipeline por paper ───────────────────────────────────────────────────────
def process_paper(row, out_dir: Path, server: str,
                  top_k: int, chunk_words: int, overlap: int,
                  max_tokens: int, temperature: float, timeout: int,
                  keep_figures: bool, extract_only: bool = False,
                  keep_pdf: bool = False) -> dict:

    pmcid     = row.pmcid
    doi       = getattr(row, "doi", "")
    title     = getattr(row, "title", "")
    text      = getattr(row, "text_clean", "")
    paper_dir = out_dir / pmcid

    status = {
        "pmcid": pmcid, "doi": doi, "title": title,
        "status": "ok", "error": None,
        "figures": 0, "tables": 0, "elapsed_sec": 0,
    }
    t0 = time.time()

    # Resume: saltar si ya completó el paso correspondiente
    done_marker = paper_dir / ("figures.json" if extract_only else "analysis.json")
    if done_marker.exists():
        print(f"  SKIP {pmcid} (ya procesado)")
        status["status"] = "skip"
        return status

    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / "paper.pdf"

    # 1) Descargar PDF
    print(f"  Descargando {pmcid} ...")
    ok = download_pdf(pmcid, doi, pdf_path)
    if not ok:
        print(f"  ERROR: no se pudo descargar {pmcid}")
        status["status"] = "download_failed"
        status["error"]  = "PDF not available"
        return status

    # 2) Extraer figuras
    print(f"  Extrayendo figuras ...")
    try:
        ef.extract_all(str(pdf_path), out_dir=str(paper_dir), dpi=200, quiet=True)
    except Exception as e:
        print(f"  ERROR extracción: {e}")
        pdf_path.unlink(missing_ok=True)
        status["status"] = "extract_failed"
        status["error"]  = str(e)
        return status

    figures_json = paper_dir / "figures.json"
    if not figures_json.exists():
        pdf_path.unlink(missing_ok=True)
        status["status"] = "no_figures"
        status["error"]  = "figures.json not created"
        return status

    n_figs = len(json.loads(figures_json.read_text())["items"])
    status["figures"] = n_figs
    print(f"  {n_figs} figuras extraídas")

    # 2b) Extraer tablas
    try:
        tables = et.extract_tables_all(str(pdf_path), out_dir=str(paper_dir), dpi=200, quiet=True)
        status["tables"] = len(tables)
        print(f"  {len(tables)} tabla(s) extraída(s)")
    except Exception as e:
        print(f"  WARN extracción tablas: {e}")
        status["tables"] = 0

    # Modo solo extracción: terminar aquí
    if extract_only:
        if not keep_pdf:
            pdf_path.unlink(missing_ok=True)
        status["elapsed_sec"] = round(time.time() - t0, 1)
        return status

    # 3) Escribir texto del DB a archivo temporal
    txt_path = paper_dir / "paper_context.txt"
    txt_path.write_text(text, encoding="utf-8")

    # 4) Análisis RAG (inference + anchored)
    print(f"  Analizando con RAG (top_k={top_k}) ...")
    analysis_path = paper_dir / "analysis.json"
    try:
        rag.analyze_all(
            figures_json  = str(figures_json),
            pdf_path      = None,
            context_file  = str(txt_path),
            server        = server,
            mode_inference= True,
            mode_anchored = True,
            chunk_words   = chunk_words,
            overlap       = overlap,
            top_k         = top_k,
            max_tokens    = max_tokens,
            temperature   = temperature,
            timeout       = timeout,
            out_path      = str(analysis_path),
        )
    except Exception as e:
        print(f"  ERROR análisis: {e}")
        status["status"] = "analysis_failed"
        status["error"]  = str(e)

    # 5) Eliminar PDF y contexto de texto
    pdf_path.unlink(missing_ok=True)
    txt_path.unlink(missing_ok=True)

    # 6) Eliminar figuras si no se quieren conservar
    if not keep_figures:
        for png in paper_dir.glob("*.png"):
            png.unlink()

    status["elapsed_sec"] = round(time.time() - t0, 1)
    return status


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Pipeline: parquet DB → descarga PDF → figuras → análisis RAG")

    # Fuentes de datos
    p.add_argument("--metadata", required=True,
                   help="Parquet de metadata (con pmcid, doi, main_field, quality_score)")
    p.add_argument("--texts", required=True,
                   help="Parquet de textos (con pmcid, text_clean)")

    # Filtros
    p.add_argument("--field",       help="Filtrar por main_field (ej: Agriculture)")
    p.add_argument("--quality-min", type=int, default=DEFAULT_QUALITY,
                   help=f"quality_score mínimo (def: {DEFAULT_QUALITY} = todos)")
    p.add_argument("--limit",       type=int, default=0,
                   help="Máximo papers a procesar (0 = todos)")
    p.add_argument("--offset",      type=int, default=0,
                   help="Saltar los primeros N papers")

    # Salida
    p.add_argument("--out-dir",  default=DEFAULT_OUT_DIR,
                   help=f"Directorio de salida (def: {DEFAULT_OUT_DIR})")
    p.add_argument("--keep-figures", action="store_true",
                   help="Conservar las imágenes extraídas (def: se conservan siempre)")

    # LLM
    p.add_argument("--server",      default=DEFAULT_SERVER)
    p.add_argument("--top-k",       type=int,   default=DEFAULT_TOP_K)
    p.add_argument("--chunk-words", type=int,   default=DEFAULT_CHUNK_W)
    p.add_argument("--overlap",     type=int,   default=DEFAULT_OVERLAP)
    p.add_argument("--max-tokens",  type=int,   default=rag.DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=rag.DEFAULT_TEMPERATURE)
    p.add_argument("--timeout",     type=int,   default=rag.DEFAULT_TIMEOUT)

    # Descarga
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                   help=f"Segundos entre descargas (def: {DEFAULT_DELAY})")
    p.add_argument("--extract-only", action="store_true",
                   help="Solo descarga y extrae figuras, sin correr el análisis LLM")
    p.add_argument("--keep-pdf", action="store_true",
                   help="Conservar el PDF después de extraer figuras")

    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline_log.jsonl"

    # Cargar papers
    df = load_papers(
        metadata_path = args.metadata,
        texts_path    = args.texts,
        field         = args.field,
        quality_min   = args.quality_min,
        limit         = args.limit,
        offset        = args.offset,
    )

    if df.empty:
        sys.exit("No hay papers con los filtros especificados.")

    # Verificar servidor LLM (solo si vamos a analizar)
    if not args.extract_only and not rag.server_health(args.server):
        sys.exit(f"Servidor LLM no responde: {args.server}")

    print(f"\nServidor: {args.server}")
    print(f"Salida:   {out_dir.resolve()}\n")

    stats = {"ok": 0, "skip": 0, "failed": 0}

    for n, row in enumerate(df.itertuples(), 1):
        pmcid = row.pmcid
        title = getattr(row, "title", "") or "(sin título)"
        print(f"\n[{n}/{len(df)}] {pmcid} — {title[:70]}")

        result = process_paper(
            row          = row,
            out_dir      = out_dir,
            server       = args.server,
            top_k        = args.top_k,
            chunk_words  = args.chunk_words,
            overlap      = args.overlap,
            max_tokens   = args.max_tokens,
            temperature  = args.temperature,
            timeout      = args.timeout,
            keep_figures = True,
            extract_only = args.extract_only,
            keep_pdf     = args.keep_pdf,
        )

        # Log incremental
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if result["status"] == "ok":
            stats["ok"] += 1
        elif result["status"] == "skip":
            stats["skip"] += 1
        else:
            stats["failed"] += 1

        # Delay entre papers (respetar rate limit PMC)
        if n < len(df):
            time.sleep(args.delay)

    print(f"\n{'='*50}")
    print(f"Completado: {stats['ok']} ok | {stats['skip']} skip | {stats['failed']} fallidos")
    print(f"Log: {log_path}")
    print(f"Salida: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
