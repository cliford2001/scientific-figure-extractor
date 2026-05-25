"""
run_analysis_batch.py — Corre analyze_figures_v2_rag sobre carpetas ya extraídas
Uso: python run_analysis_batch.py --input-dir sample15_output --server http://127.0.0.1:8080/v1/chat/completions
"""
from __future__ import annotations
import argparse, json, sys, time, io
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import analyze_figures_v2_rag as rag


_log_file = None

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


def main(argv=None):
    global _log_file
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="sample15_output")
    p.add_argument("--server",    default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--top-k",     type=int,   default=5)
    p.add_argument("--max-tokens",type=int,   default=rag.DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature",type=float,default=rag.DEFAULT_TEMPERATURE)
    p.add_argument("--timeout",   type=int,   default=rag.DEFAULT_TIMEOUT)
    p.add_argument("--log-file",  default="E:/pipeline/analysis_log.txt")
    args = p.parse_args(argv)
    _log_file = open(args.log_file, "w", encoding="utf-8")

    base = Path(args.input_dir)
    papers = sorted(p for p in base.iterdir() if p.is_dir())

    if not rag.server_health(args.server):
        log(f"Servidor LLM no responde: {args.server}")
        sys.exit(1)

    total_ok = total_skip = total_err = 0

    for paper_dir in papers:
        figs_json = paper_dir / "figures.json"
        tbls_json = paper_dir / "tables.json"
        ctx_file  = paper_dir / "paper_context.txt"
        ana_json  = paper_dir / "analysis.json"

        if not figs_json.exists():
            continue

        n_figs = len(json.loads(figs_json.read_text())["items"])
        n_tbls = len(json.loads(tbls_json.read_text())["items"]) if tbls_json.exists() else 0
        if n_figs == 0 and n_tbls == 0:
            log(f"[SKIP] {paper_dir.name} — 0 figuras y 0 tablas")
            total_skip += 1
            continue

        if ana_json.exists():
            log(f"[SKIP] {paper_dir.name} — ya analizado")
            total_skip += 1
            continue

        log(f"[->] {paper_dir.name} ({n_figs} figs + {n_tbls} tablas) ...")
        t0 = time.time()
        try:
            rag.analyze_all(
                figures_json   = str(figs_json),
                tables_json    = str(tbls_json) if tbls_json.exists() else None,
                pdf_path       = None,
                context_file   = str(ctx_file) if ctx_file.exists() else None,
                server         = args.server,
                mode_inference = True,
                mode_anchored  = True,
                top_k          = args.top_k,
                max_tokens     = args.max_tokens,
                temperature    = args.temperature,
                timeout        = args.timeout,
                out_path       = str(ana_json),
            )
            elapsed = round(time.time() - t0, 1)
            log(f"    OK — {elapsed}s -> {ana_json.name}")
            total_ok += 1
        except Exception as e:
            log(f"    ERROR: {e}")
            total_err += 1

    log(f"{'='*50}")
    log(f"OK: {total_ok} | Skip: {total_skip} | Error: {total_err}")
    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()
