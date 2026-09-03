# Repository Guidelines

## Agent-Specific Instructions

Use Chinese when communicating with the user.

## Project Structure & Module Organization

This repository is currently in the M0 necessity-study phase. Keep the research boundary and acceptance criteria in `docs/research-contract.md`. Paper sources belong in `paper/` (the entry point is `paper/main.tex`), mechanism code in `src/`, reproducible experiment entry points in `scripts/`, and result metadata in `results/`. The planned ECHO runtime should be pinned as `runtime/ECHO`, with workloads and analysis code placed in `workloads/` and `analysis/` as they are introduced. Do not add a memory-aware predictor before the M0 gate is satisfied.

## Build, Test, and Development Commands

There is no repository-wide build or test runner yet. For the current tree, use:

```bash
latexmk -pdf -cd paper/main.tex       # build the paper
latexmk -C -cd paper/main.tex         # remove generated LaTeX files
git submodule update --init --recursive  # restore the pinned runtime once added
```

When adding implementation code, document exact setup and execution commands in the nearest README. Experiment scripts must capture runtime/model revisions, hardware, workload, memory budget, concurrency, seed, warmup, repetitions, and raw-output location.

## Coding Style & Naming Conventions

Follow the conventions of the language and the pinned runtime. Use four spaces and `snake_case` for Python functions, modules, and experiment scripts; use `PascalCase` for classes. Prefer descriptive script names such as `run_m0_no_prefetch.py`. Keep shell entry points non-interactive and fail fast. Format Markdown with concise headings and fenced command examples. Never modify the native sparse-attention indexer's final selected KV set; memory signals may affect only prefetch timing, residency, and recall scheduling.

## Testing & Evidence Guidelines

No automated test framework or coverage threshold is configured. New mechanism code should include focused unit tests plus matched end-to-end checks. Verify exact selected-block and output equivalence, and compare baselines under identical model, trace, concurrency, KV budget, sampling, and hardware placement. Smoke-only, projected, synthetic, or unmatched measurements do not belong in the paper evidence set. Record accepted runs in `results/MANIFEST.md`; keep bulky raw outputs outside Git and reference their stable location.

## Commit & Pull Request Guidelines

The history currently uses short, imperative subjects (for example, `Initialize memory-aware sparse KV research topic`). Keep commits focused and avoid mixing instrumentation, mechanism, and paper claims. Pull requests should explain the research question, changed paths, reproduction commands, runtime/model commit IDs, and validation results. Link the relevant issue; include plots or log excerpts when results or performance claims change, and identify limitations or unmatched conditions explicitly.
