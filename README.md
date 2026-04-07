# fdga-chain

GerryChain ensemble analysis for Georgia redistricting education.
Part of the [Fair Districts GA](https://fairdistrictsga.org) toolset.

## Purpose

Generates ensembles of thousands of alternative redistricting maps that satisfy
legal requirements (equal population, contiguous districts), then compares the
enacted map against this neutral baseline. Used to educate local officials and
volunteers on the mathematical effects of gerrymandering — non-partisan and
evidence-based.

## Related Projects

| Project | Purpose |
|---|---|
| **lrdb** | Local Redistricting Database — tracks local government redistricting status |
| **fdex** | Consumer-facing district explorer (maps, election overlays) |
| **fdga-chain** | Ensemble analysis and education tools (this project) |

## Setup

```bash
uv sync
cp .env.example .env
```

## Workflow

### 1. Get precinct data (one-time)

```bash
uv run python scripts/fetch_precinct_data.py
# Follow instructions to download GA precinct shapefile
```

### 2. Build dual graph (one-time per shapefile)

```bash
uv run python scripts/build_graph.py --chamber house
uv run python scripts/build_graph.py --chamber senate
uv run python scripts/build_graph.py --chamber congress
```

Output: `data/graphs/ga_{chamber}.json`

### 3. Run ensemble (computationally intensive, run offline)

```bash
uv run python scripts/run_ensemble.py --chamber house --steps 10000
uv run python scripts/run_ensemble.py --chamber senate --steps 5000
```

Takes 30-60 minutes on a laptop. Output: `data/ensembles/{chamber}_ensemble.parquet`

### 4. Start the API server

```bash
uv run uvicorn api.main:app --reload --port 8001
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/chambers` | List chambers with ready ensembles |
| `GET /api/ensemble/{chamber}/summary` | Full distribution stats for all metrics |
| `GET /api/ensemble/{chamber}/histogram?metric=dem_seats` | Histogram data |
| `GET /api/ensemble/{chamber}/enacted` | Enacted plan vs. ensemble with plain-English interpretation |
| `GET /api/lrdb/jurisdiction/{name}` | Look up a jurisdiction in LRDB |

## Metrics Computed

- **dem_seats** — Democratic seat count per plan
- **efficiency_gap** — Wasted-vote asymmetry (>5% = significant partisan bias)
- **mean_median** — Mean vs. median Democratic vote share
- **polsby_popper_mean/min** — District compactness (0-1, higher = more compact)
- **majority_minority_districts** — Districts where minority VAP >= 50%
- **num_cut_edges** — Map fragmentation proxy

## Education Use

The key output is `/api/ensemble/{chamber}/enacted` — for each metric it returns:

```json
{
  "dem_seats": {
    "enacted_value": 65,
    "ensemble_median": 72.3,
    "percentile_rank": 4.2,
    "is_outlier": true,
    "interpretation": "Democrats win 7.3 fewer seats than the median neutral map."
  }
}
```

This is designed to be shown directly to local officials: "Of 10,000 randomly
drawn fair maps, only 4% produced fewer Democratic seats than the current map."
