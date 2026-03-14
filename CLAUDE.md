# CLAUDE.md — Project Instructions

## Project
**Flash Flood Risk Intelligence for Himachal Pradesh**
Goal: A decision tool for HP SDMA / PWD / tourism board showing which roads, villages,
and valleys face highest flash flood risk — and why. Paper wraps around the tool.
See `.claude/PLAN.md` for the full implementation plan.

## Output
- **Primary:** A web-based risk dashboard (interactive map + district briefings)
- **Secondary:** A peer-reviewed journal paper (NHESS or Remote Sensing)
- **Novelty:** GNN on watershed graph + Conformal Prediction uncertainty + SAR inventory

## Repository Structure
```
flash-flood-zones-hp/        ← monorepo root
├── apps/
│   └── web/                 # Next.js 15 dashboard (risk map + briefings)
├── packages/
│   ├── ui/                  # shadcn/ui component library
│   └── api/                 # tRPC routes (serve ML results to frontend)
├── scripts/                 # Python ML pipeline (numbered execution order)
├── data/
│   ├── raw/                 # original rasters, shapefiles (gitignored)
│   └── processed/           # cleaned, clipped, derived layers
├── results/                 # model outputs, susceptibility rasters (gitignored)
├── literature/markdown/     # papers as markdown
├── paper/                   # journal paper (LaTeX or markdown)
└── .claude/PLAN.md          # implementation plan + progress tracker
```

## Tech Stack
### Dashboard (apps/web)
- **Framework:** Next.js 15 (App Router)
- **UI:** shadcn/ui + Tailwind CSS
- **Map:** MapLibre GL JS or react-map-gl (Mapbox-compatible, free)
- **Charts:** Recharts or Tremor
- **State:** Zustand + TanStack Query
- **API:** tRPC
- **Language:** TypeScript

### ML Pipeline (scripts/)
- **Language:** Python 3.11+ via uv
- **Core ML:** scikit-learn, XGBoost, LightGBM, PyTorch Geometric (GNN)
- **Uncertainty:** MAPIE (conformal prediction)
- **Geospatial:** rasterio, geopandas, shapely, pyproj
- **SAR processing:** Google Earth Engine Python API
- **Explainability:** SHAP
- **Viz:** matplotlib, seaborn

### Monorepo
- **Package manager:** pnpm workspaces
- **Build system:** Turborepo
- **Linting:** Biome
- **TypeScript:** strict mode

## Commands
### JS/TS (dashboard)
- `pnpm dev` — start all apps
- `pnpm build` — build all
- `pnpm check` — Biome lint + format
- `pnpm ui-add` — add shadcn component

### Python (ML pipeline)
- `uv sync` — install Python deps
- `uv run python scripts/<script>.py` — run a script
- `uv run jupyter lab` — open notebooks

## Commit Rules
- **No Claude email** in commits — no `Co-Authored-By` lines
- **Layered commits** — one logical change per commit
- Commit messages: concise and descriptive
- kebab-case for all file and directory names

## Code Conventions
- TypeScript: always `import type` for types; no `any`
- Python: config-driven, no hardcoded paths
- Raw data is never modified — always work on copies
- All ML outputs go in `results/`
- All scripts go in `scripts/`
