# scientific-figure-extractor

Multimodal pipeline for automated analysis of figures and tables in scientific PDFs. Extracts every figure and table as an isolated PNG, then analyzes each one with a vision-language model using two complementary modes — pure visual inference and RAG-grounded analysis — producing structured JSON output suitable for LLM training data.

---

## Overview

```
SQLite DB ──► Download PDF ──► Extract ──────────────────► Analyze LLM ──► analyses_rag.json
(DOI, text)                    figures.json                 ↑
                               tables.json    paper_context.txt
                               paper_context.txt  (abstract detection + BM25 / full-text)
```

**Two analysis modes per figure/table:**

| Mode | Input | What it produces |
|---|---|---|
| **Inference** | Image + Caption + Abstract | What the model reads purely from the image |
| **Anchored** | Image + Caption + Abstract + Paper text (BM25 or full) | Analysis grounded in the author's own text |

The divergence between both modes is where the analytical value lies: inference reveals what is visually evident; anchored reveals what the authors claim and whether the image supports it.

---

## Scripts

| Script | Purpose |
|---|---|
| `extract_figures.py` | Extracts figures from a PDF → `figures.json` + PNGs |
| `extract_tables.py` | Extracts tables from a PDF → `tables.json` + PNGs |
| `analyze_figures_v2_rag.py` | Analyzes figures and tables with a VLM + RAG context |
| `run_analysis_batch.py` | Batch runner over a folder of papers |
| `pipeline_db_to_analysis.py` | End-to-end from SQLite DB → extraction → analysis |

---

## Step 1 — Extract figures

```bash
python extract_figures.py paper.pdf --out extracted/ --dpi 200
```

**How it works:**
1. Detects captions via regex: `Figure N`, `Fig. N`, `Extended Data Fig. N`
2. Classifies caption column (left / right / full-width) from horizontal position
3. Finds visual content above the caption in the same column
4. Expands the bounding box to capture axis labels, panel letters, colorbars (within 60 pt)
5. Filters body text wider than 35% of page width (avoids capturing paragraphs)
6. Handles cross-page figures (caption on page N, visual on page N+1)
7. Falls back to pixel rendering for embedded Form XObjects
8. Renders final bbox to PNG at configurable DPI

No ML. Runs on CPU in milliseconds per page.

**Output:**
```
extracted/
├── p002_fig_1.png
├── p003_fig_2.png
└── figures.json
```

```json
{
  "pdf": "paper.pdf",
  "total": 8,
  "items": [
    {
      "label":      "fig_1",
      "kind":       "figure",
      "page":       2,
      "caption":    "Figure 1. Glucose uptake...",
      "image_path": "extracted/p002_fig_1.png",
      "image_size": [640, 420]
    }
  ]
}
```

**Options:**
```
--out DIR        output directory (default: extracted/)
--dpi N          render resolution (default: 200)
--max-height N   max figure height in points (default: 9999)
--quiet          suppress stdout
```

---

## Step 2 — Extract tables

```bash
python extract_tables.py paper.pdf --out extracted/
```

**How it works:**
1. `find_tables()` with `lines_strict` strategy (PyMuPDF)
2. Camelot lattice as fallback for complex grid tables
3. Area filter: minimum 5 000 pt² to skip tiny inline tables
4. Merges "Continued" tables across pages
5. Renders each table region to PNG

**Output:** `tables.json` with the same schema as `figures.json` (`kind: "table"`).

---

## Step 3 — Analyze (single paper)

```bash
# BM25 mode — default, works on any GPU
python analyze_figures_v2_rag.py extracted/figures.json \
    --context-file extracted/paper_context.txt \
    --server http://127.0.0.1:8080/v1/chat/completions

# Full-text mode — requires large context window (≥ 32 768 tokens)
python analyze_figures_v2_rag.py extracted/figures.json \
    --context-file extracted/paper_context.txt \
    --context-strategy full

# Include tables
python analyze_figures_v2_rag.py extracted/figures.json \
    --tables-json extracted/tables.json \
    --context-file extracted/paper_context.txt
```

### Context strategies

#### BM25 (default — for testing and small GPUs)

Splits the paper into 250-word chunks with 40-word overlap, then for each figure retrieves the top-10 most relevant chunks using BM25 ranking (the caption is used as the query). Based on [Lewis et al., NeurIPS 2020](https://arxiv.org/abs/2005.11401) — DOI: 10.48550/arXiv.2005.11401.

```
paper_context.txt (10 000+ words)
  ↓ split into 250-word chunks
[chunk_1] [chunk_2] ... [chunk_N]
  ↓ BM25 search (query = caption)
top-10 most relevant chunks
  ↓ injected into anchored prompt
```

#### Full-text (for large GPUs)

Passes the complete paper text to the LLM without retrieval. Always superior to BM25 when the GPU context window allows it.

| GPU | Context window | Recommended setting |
|---|---|---|
| GTX 1080 Ti (11 GB) | 12 288 tokens | `--context-strategy bm25` or `--context-strategy full --max-context-words 3000` |
| RTX 3090 / 4090 | 32 768 tokens | `--context-strategy full` |
| A100 / H100 | 128 000+ tokens | `--context-strategy full` |

### Abstract detection

Instead of a fixed word count, the pipeline detects the real abstract section by searching for the `Abstract` heading and cutting at the next section heading (`Introduction`, `Methods`, `Results`, `Keywords`, `1.`, etc.). The detected abstract is injected into **all** prompts (both inference and anchored). Falls back to the first 400 words if no heading is found.

### Prompts

**Inference — figure output fields:**

| Field | Description |
|---|---|
| `figure_type` | Chart type: bar chart, scatter, Western blot, microscopy, heatmap, etc. |
| `visual_description` | Panels, axes, colors, units, structures visible. 2-3 sentences. |
| `groups_compared` | Conditions, treatments, timepoints or genotypes contrasted. |
| `statistical_markers` | Every visible statistical element: n=, error bars (SD/SEM/CI), p-values, R², fold-changes. |
| `key_finding` | Main result with specific numbers read from the image. |
| `caption_accurate` | `true`/`false` — does the caption correctly describe what is shown? |
| `scientific_interpretation` | What biological question this figure addresses and what the data demonstrates. |
| `confidence` | `high` = clearly readable · `medium` = partially visible · `low` = inferring |

**Anchored — additional fields over inference:**

| Field | Description |
|---|---|
| `hypothesis_tested` | Specific claim from the paper this figure tests. Quotes the relevant sentence. |
| `paper_quote` | Exact sentence from the paper text this figure is meant to support. |
| `controls_assessment` | Experimental controls present. Notes absent controls given the design. |
| `scientific_conclusion` | What this figure definitively demonstrates, what alternatives it rules out, its role in the paper's argument. |

**Tables use equivalent fields:** `table_type`, `structure`, `best_result`, `key_pattern`, `key_entries`.

> `confidence` is an LLM self-assessment — useful as metadata for human review, not as an automatic filter.

### Paper-level synthesis

After all figures and tables are analyzed, one additional text-only LLM call reads all individual findings and generates a `paper_summary` block:

| Field | Description |
|---|---|
| `main_contribution` | Central claim of the paper in 1-2 sentences. |
| `narrative` | How figures and tables build the paper's argument step by step. |
| `key_evidence` | Labels of the 3 most critical items. |
| `contradictions_or_gaps` | Figures that contradict each other or gaps in evidence. |
| `limitations_noted` | Methodological or statistical limitations visible across analyses. |
| `overall_confidence` | `high` / `medium` / `low` |

### Full output structure

```json
{
  "total": 12,
  "context_strategy": "bm25",
  "paper_summary": {
    "main_contribution": "...",
    "narrative": "Fig1 establishes baseline → Fig2 demonstrates mechanism → Table1 validates statistically...",
    "key_evidence": ["fig_1", "fig_3", "table_1"],
    "contradictions_or_gaps": "None detected",
    "limitations_noted": "n=24 per group, no placebo control in Fig4",
    "overall_confidence": "high"
  },
  "items": [
    {
      "label": "fig_1",
      "kind":  "figure",
      "page":  3,
      "inference_parsed": {
        "key_finding": "34% glucose reduction (p<0.001)",
        "confidence":  "high"
      },
      "anchored_parsed": {
        "paper_quote":           "We observed a significant reduction...",
        "scientific_conclusion": "..."
      },
      "confidence":   "high",
      "elapsed_sec":  42.3
    }
  ]
}
```

### All options

```
--context-strategy    "bm25" (default) | "full"
--max-context-words   max words in full-text mode (0 = no limit)
--abstract-words      words from abstract (0 = full detected section)
--top-k               BM25 chunks per item (default: 10)
--chunk-words         words per BM25 chunk (default: 250)
--chunk-overlap       overlap between chunks (default: 40)
--max-tokens          LLM response tokens (default: 1500)
--temperature         sampling temperature (default: 0.0)
--timeout             seconds per request (default: 300)
--inference-only      skip anchored mode
--anchored-only       skip inference mode
--out FILE            output path (default: analyses_rag.json)
```

Resumes automatically if interrupted — already-completed items are skipped.

---

## Step 4 — Batch over multiple papers

```bash
# Testing on GTX 1080 Ti (BM25, default)
python run_analysis_batch.py --input-dir sample15_output

# Production on large GPU (full-text, no limit)
python run_analysis_batch.py --input-dir sample15_output \
    --context-strategy full

# GTX 1080 Ti with truncated full-text
python run_analysis_batch.py --input-dir sample15_output \
    --context-strategy full --max-context-words 3000
```

**Expected folder structure:**
```
sample15_output/
├── paper_001/
│   ├── figures.json
│   ├── tables.json          ← optional
│   ├── paper_context.txt    ← required for anchored mode
│   └── analyses_rag.json    ← generated output
├── paper_002/
│   └── ...
```

**Skip logic:**
- Skips paper if `analyses_rag.json` already has all items complete + `paper_summary`
- Resumes partial analyses automatically (per-item resume inside `analyze_all`)
- Falls back to inference-only if `paper_context.txt` is missing
- Use `--rerun` to force re-analysis of already-complete papers

**All batch options:**
```
--input-dir           root folder with one subfolder per paper
--server              llama.cpp endpoint (default: http://127.0.0.1:8080/v1/chat/completions)
--log-file            path for append-mode log file
--context-strategy    "bm25" | "full"
--max-context-words   truncation limit for full-text mode
--abstract-words      0 = full detected abstract
--top-k               BM25 chunks (default: 10)
--chunk-words         BM25 chunk size (default: 250)
--chunk-overlap       BM25 overlap (default: 40)
--max-tokens          LLM response tokens
--temperature         sampling temperature
--timeout             seconds per request
--inference-only      skip anchored mode for all papers
--anchored-only       skip inference mode for all papers
--rerun               ignore existing analyses_rag.json
```

---

## LLM server setup (llama.cpp)

```bash
# GTX 1080 Ti — 8B quantized model, context 12 288
llama-server \
    -m InternVL3-8B-Q8_0.gguf \
    --mmproj mmproj-InternVL3-8B-Q8_0.gguf \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 -np 1 \
    -c 12288 \
    --jinja

# Large GPU — full context, no limit
llama-server \
    -m InternVL3-14B-Q4_K_M.gguf \
    --mmproj mmproj-InternVL3-14B-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 \
    -c 32768 \
    --jinja
```

**Tested models (GGUF):**
- [ggml-org/InternVL3-8B-Instruct-GGUF](https://huggingface.co/ggml-org/InternVL3-8B-Instruct-GGUF)
- [ggml-org/InternVL3-14B-Instruct-GGUF](https://huggingface.co/ggml-org/InternVL3-14B-Instruct-GGUF)

---

## Token budget estimates

| Component | Inference fig | Inference tbl | Anchored BM25 fig | Anchored BM25 tbl |
|---|---|---|---|---|
| Prompt template | ~365 | ~310 | ~420 | ~380 |
| Abstract (full section) | ~500–1 000 | ~500–1 000 | ~500–1 000 | ~500–1 000 |
| BM25 context (10 × 250w) | — | — | ~3 250 | ~3 250 |
| Caption | ~65 | ~65 | ~65 | ~65 |
| Image PNG | ~500–1 000 | ~300–600 | ~500–1 000 | ~300–600 |
| Response (max_tokens) | 1 500 | 1 500 | 1 500 | 1 500 |
| **Total** | **~3–4K** | **~2.7–3.5K** | **~6.2–8K** | **~6–7.8K** |

GTX 1080 Ti (ctx 12 288): comfortable with BM25. For full-text mode use `--max-context-words 3000`.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- PyMuPDF >= 1.24
- requests >= 2.31
- camelot-py (optional, for complex table extraction)
- rank-bm25 (optional, automatic fallback to TF-IDF if not installed)

---

## Known limitations

- **Paywall papers (HTTP 403):** Papers behind MDPI, Wiley, Taylor & Francis paywalls cannot be downloaded without institutional access. The pipeline skips them automatically.
- **Caption position:** Assumes captions appear below the figure (standard in most journals). Captions above figures are not detected.
- **BM25 retrieval:** Lexical matching only — misses semantically related text that uses different terminology. Use full-text mode when GPU allows.
- **LLM self-assessment:** `confidence` is generated by the model itself and should be treated as informational metadata, not a reliable filter.

---

## References

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)

---

## License

MIT — see [LICENSE](LICENSE).
