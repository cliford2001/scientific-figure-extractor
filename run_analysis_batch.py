"""
run_analysis_batch.py — Corre analyze_figures_v2_rag sobre carpetas ya extraídas.

Uso básico (testeo en 1080 Ti, BM25 por defecto):
    python run_analysis_batch.py --input-dir sample15_output

Producción en GPU grande (texto completo sin límite):
    python run_analysis_batch.py --input-dir sample15_output --context-strategy full

1080 Ti con texto completo truncado a 3000 palabras:
    python run_analysis_batch.py --input-dir sample15_output --context-strategy full --max-context-words 3000
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_figures_v2_rag as rag


_log_file = None

def log(msg: str):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


def main(argv=None):
    global _log_file
    p = argparse.ArgumentParser(
        description="Batch de análisis LLM sobre figuras y tablas extraídas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rutas
    p.add_argument("--input-dir",  default="sample15_output",
                   help="Carpeta raíz con subcarpetas por paper")
    p.add_argument("--server",     default=rag.DEFAULT_SERVER,
                   help="URL del servidor llama.cpp")
    p.add_argument("--log-file",   default="E:/pipeline/analysis_log.txt",
                   help="Ruta del archivo de log")

    # Estrategia de contexto
    p.add_argument("--context-strategy",  choices=["bm25", "full"],
                   default=rag.DEFAULT_CONTEXT_STRATEGY,
                   help="'bm25' = top-k chunks (testeo) | 'full' = texto completo (GPU grande)")
    p.add_argument("--max-context-words", type=int,
                   default=rag.DEFAULT_MAX_CONTEXT_WORDS,
                   help="Palabras máx en modo full (0 = sin límite)")
    p.add_argument("--abstract-words",    type=int,
                   default=rag.DEFAULT_ABSTRACT_WORDS,
                   help="Palabras del abstract a inyectar (0 = abstract completo detectado)")

    # Parámetros BM25 (solo aplican con --context-strategy bm25)
    p.add_argument("--top-k",        type=int,   default=rag.DEFAULT_TOP_K,
                   help="Fragmentos BM25 recuperados por ítem")
    p.add_argument("--chunk-words",  type=int,   default=rag.DEFAULT_CHUNK_WORDS,
                   help="Palabras por fragmento BM25")
    p.add_argument("--chunk-overlap",type=int,   default=rag.DEFAULT_CHUNK_OVERLAP,
                   help="Superposición entre fragmentos BM25")

    # LLM
    p.add_argument("--max-tokens",   type=int,   default=rag.DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature",  type=float, default=rag.DEFAULT_TEMPERATURE)
    p.add_argument("--timeout",      type=int,   default=rag.DEFAULT_TIMEOUT)

    # Modos
    p.add_argument("--inference-only", action="store_true",
                   help="Solo modo inferencia (sin contexto RAG)")
    p.add_argument("--anchored-only",  action="store_true",
                   help="Solo modo anclado (requiere contexto)")
    p.add_argument("--rerun",          action="store_true",
                   help="Ignorar analyses_rag.json existente y re-analizar")

    args = p.parse_args(argv)

    try:
        _log_file = open(args.log_file, "a", encoding="utf-8")
    except Exception as e:
        print(f"WARN: no se pudo abrir log file: {e}")

    log("=" * 60)
    log(f"Iniciando batch — input: {args.input_dir}")
    log(f"Estrategia: {args.context_strategy} | abstract_words: {args.abstract_words or 'completo'}")
    if args.context_strategy == "bm25":
        log(f"BM25: top_k={args.top_k} | chunk_words={args.chunk_words} | overlap={args.chunk_overlap}")
    if args.context_strategy == "full" and args.max_context_words:
        log(f"Full-text: max_context_words={args.max_context_words}")
    log("=" * 60)

    if not rag.server_health(args.server):
        log(f"ERROR: Servidor LLM no responde: {args.server}")
        sys.exit(1)

    base   = Path(args.input_dir)
    papers = sorted(d for d in base.iterdir() if d.is_dir())

    mode_inf = not args.anchored_only
    mode_anc = not args.inference_only

    total_ok = total_skip = total_err = 0

    for paper_dir in papers:
        figs_json = paper_dir / "figures.json"
        tbls_json = paper_dir / "tables.json"
        ctx_file  = paper_dir / "paper_context.txt"
        ana_json  = paper_dir / "analyses_rag.json"

        if not figs_json.exists():
            log(f"[SKIP] {paper_dir.name} — sin figures.json")
            total_skip += 1
            continue

        n_figs = len(json.loads(figs_json.read_text(encoding="utf-8"))["items"])
        n_tbls = (len(json.loads(tbls_json.read_text(encoding="utf-8"))["items"])
                  if tbls_json.exists() else 0)

        if n_figs == 0 and n_tbls == 0:
            log(f"[SKIP] {paper_dir.name} — 0 figuras y 0 tablas")
            total_skip += 1
            continue

        if ana_json.exists() and not args.rerun:
            try:
                prev    = json.loads(ana_json.read_text(encoding="utf-8"))
                n_done  = prev.get("total", 0)
                n_total = n_figs + n_tbls
                tiene_sintesis = prev.get("paper_summary") is not None
                if n_done >= n_total and tiene_sintesis:
                    log(f"[SKIP] {paper_dir.name} — ya completo ({n_done} ítems + síntesis)")
                    total_skip += 1
                    continue
                else:
                    log(f"[->] {paper_dir.name} — retomando ({n_done}/{n_total} ítems)")
            except Exception:
                pass
        else:
            log(f"[->] {paper_dir.name} ({n_figs} figs + {n_tbls} tablas) ...")

        if mode_anc and not ctx_file.exists():
            log(f"  WARN: sin paper_context.txt — solo modo inferencia")
            mode_anc_efectivo = False
        else:
            mode_anc_efectivo = mode_anc

        t0 = time.time()
        try:
            rag.analyze_all(
                figures_json      = str(figs_json),
                tables_json       = str(tbls_json) if tbls_json.exists() else None,
                pdf_path          = None,
                context_file      = str(ctx_file) if ctx_file.exists() else None,
                server            = args.server,
                mode_inference    = mode_inf,
                mode_anchored     = mode_anc_efectivo,
                context_strategy  = args.context_strategy,
                max_context_words = args.max_context_words,
                abstract_words    = args.abstract_words,
                top_k             = args.top_k,
                chunk_words       = args.chunk_words,
                overlap           = args.chunk_overlap,
                max_tokens        = args.max_tokens,
                temperature       = args.temperature,
                timeout           = args.timeout,
                out_path          = str(ana_json),
            )
            elapsed = round(time.time() - t0, 1)
            log(f"    OK — {elapsed}s → {ana_json.name}")
            total_ok += 1
        except Exception as e:
            log(f"    ERROR: {e}")
            total_err += 1

    log("=" * 60)
    log(f"Finalizado — OK: {total_ok} | Saltados: {total_skip} | Errores: {total_err}")
    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()
