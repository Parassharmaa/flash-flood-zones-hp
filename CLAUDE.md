# CLAUDE.md — Project Instructions

## Project
Flash Flood Zone Mapping and Risk Analysis in Himachal Pradesh.
See `.claude/PLAN.md` for the full implementation plan.

## Commands
- `uv sync` — install dependencies
- `uv run python scripts/<script>.py` — run a script
- `uv run jupyter lab` — open notebooks

## Project Structure
```
scripts/           # analysis scripts (numbered in execution order)
results/           # all outputs (gitignored large files)
data/
  raw/             # original data, never modified
  processed/       # cleaned/derived datasets
literature/        # papers (pdf + markdown)
paper/             # final report / thesis
.claude/           # plan, memory
```

## Commit Rules
- **No Claude email** in commit messages — no `Co-Authored-By` lines
- **Layered commits** — one logical change per commit, never bundle unrelated changes
- Commit messages should be concise and descriptive

## Code Conventions
- Python (primary)
- Use `uv` for dependency management
- Use latest stable library versions
- Config-driven where possible — no hardcoded paths or hyperparameters
- Raw data is never modified — always work on copies
- All scripts go in `scripts/`
- All outputs go in `results/`

## Key Files
- `.claude/PLAN.md` — full implementation plan with progress tracker
- `data/raw/` — original rasters, shapefiles, CSVs (never modify)
- `data/processed/` — cleaned, clipped, reprojected layers
- `results/` — model outputs, maps, tables
- `literature/markdown/` — papers converted to markdown for reference
