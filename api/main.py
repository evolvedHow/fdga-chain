"""
fdga-chain API — serves pre-computed ensemble results for the education UI.

Endpoints:
  GET /api/chambers                        — list available chambers
  GET /api/ensemble/{chamber}/summary      — distribution stats for all metrics
  GET /api/ensemble/{chamber}/histogram    — histogram data for a metric
  GET /api/ensemble/{chamber}/enacted      — enacted plan metrics + percentiles
  GET /api/districts/{chamber}             — per-district stats from enacted plan
  GET /api/lrdb/jurisdiction/{name}        — look up a jurisdiction from lrdb data
  GET /health

Run:
  uv run uvicorn api.main:app --reload --port 8001
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="fdga-chain", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

ENSEMBLE_DIR = Path("data/ensembles")
FRONTEND_DIR = Path("frontend")
RAW_DIR      = Path("data/raw")
Chamber = Literal["house", "senate", "congress"]

DISTRICT_SHAPEFILES = {
    "house":    RAW_DIR / "House-2023 shape.shp",
    "senate":   RAW_DIR / "Senate-2023 shape file.shp",
    "congress": RAW_DIR / "Congress-2023 shape.shp",
}

# In-memory cache for GeoJSON (loaded once, re-used)
_geo_cache: dict = {}

METRICS = ["dem_seats", "efficiency_gap", "mean_median",
           "polsby_popper_mean", "polsby_popper_min",
           "majority_minority_districts", "num_cut_edges"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ensemble(chamber: str) -> tuple[pd.DataFrame, dict]:
    parquet = ENSEMBLE_DIR / f"{chamber}_ensemble.parquet"
    meta_path = ENSEMBLE_DIR / f"{chamber}_meta.json"
    if not parquet.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No ensemble data for chamber '{chamber}'. "
                   f"Run: uv run python scripts/run_ensemble.py --chamber {chamber}"
        )
    df = pd.read_parquet(parquet)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return df, meta


def _percentile(series: pd.Series, value: float) -> float:
    return float((series <= value).mean() * 100)


def _hist(series: pd.Series, bins: int = 30) -> dict:
    counts, edges = np.histogram(series.dropna(), bins=bins)
    return {
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    available = [
        p.stem.replace("_ensemble", "")
        for p in ENSEMBLE_DIR.glob("*_ensemble.parquet")
    ]
    return {"status": "ok", "chambers_ready": available}


@app.get("/api/chambers")
def list_chambers():
    """Return which chambers have pre-computed ensembles available."""
    results = []
    for chamber in ["house", "senate", "congress"]:
        meta_path = ENSEMBLE_DIR / f"{chamber}_meta.json"
        parquet = ENSEMBLE_DIR / f"{chamber}_ensemble.parquet"
        if parquet.exists():
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            results.append({
                "chamber": chamber,
                "steps": meta.get("steps"),
                "num_districts": meta.get("num_districts"),
                "runtime_seconds": meta.get("runtime_seconds"),
            })
    return results


@app.get("/api/ensemble/{chamber}/summary")
def ensemble_summary(chamber: Chamber):
    """
    Distribution summary for every metric: mean, median, std, p5, p25, p75, p95.
    Also includes enacted plan percentile rank for each metric.
    """
    df, meta = _load_ensemble(chamber)
    enacted = meta.get("enacted_metrics", {})

    summary = {}
    for col in METRICS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        entry = {
            "mean":   round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std":    round(float(s.std()), 4),
            "p5":     round(float(s.quantile(0.05)), 4),
            "p25":    round(float(s.quantile(0.25)), 4),
            "p75":    round(float(s.quantile(0.75)), 4),
            "p95":    round(float(s.quantile(0.95)), 4),
            "min":    round(float(s.min()), 4),
            "max":    round(float(s.max()), 4),
        }
        if col in enacted:
            entry["enacted_value"] = round(float(enacted[col]), 4)
            entry["enacted_percentile"] = round(_percentile(s, enacted[col]), 1)
        summary[col] = entry

    return {
        "chamber": chamber,
        "total_plans": len(df),
        "num_districts": meta.get("num_districts"),
        "epsilon": meta.get("epsilon"),
        "metrics": summary,
    }


@app.get("/api/ensemble/{chamber}/histogram")
def ensemble_histogram(
    chamber: Chamber,
    metric: str = "dem_seats",
    bins: int = 30,
):
    """Histogram data for a single metric, suitable for bar charts."""
    if metric not in METRICS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown metric. Choose from: {METRICS}")
    df, meta = _load_ensemble(chamber)
    if metric not in df.columns:
        raise HTTPException(status_code=404,
                            detail=f"Metric '{metric}' not in ensemble data")

    enacted = meta.get("enacted_metrics", {})
    hist = _hist(df[metric], bins=bins)
    result = {
        "chamber": chamber,
        "metric": metric,
        "histogram": hist,
        "total_plans": len(df),
    }
    if metric in enacted:
        result["enacted_value"] = enacted[metric]
        result["enacted_percentile"] = round(
            _percentile(df[metric].dropna(), enacted[metric]), 1
        )
    return result


@app.get("/api/ensemble/{chamber}/enacted")
def enacted_comparison(chamber: Chamber):
    """
    Enacted plan metrics with full percentile context.
    This is the primary education endpoint — "where does our map fall?"
    """
    df, meta = _load_ensemble(chamber)
    enacted = meta.get("enacted_metrics", {})

    comparison = {}
    for col in METRICS:
        if col not in df.columns or col not in enacted:
            continue
        s = df[col].dropna()
        val = enacted[col]
        pct = _percentile(s, val)

        # Plain-English interpretation for education UI
        if col == "efficiency_gap":
            if val > 0.05:
                interp = f"This map wastes {abs(val)*100:.1f}% more Democratic votes — favors Republicans."
            elif val < -0.05:
                interp = f"This map wastes {abs(val)*100:.1f}% more Republican votes — favors Democrats."
            else:
                interp = "Efficiency gap is within the ±5% competitive range."
        elif col == "dem_seats":
            median_seats = float(s.median())
            diff = val - median_seats
            if abs(diff) < 0.5:
                interp = "Seat share matches what neutral maps typically produce."
            elif diff > 0:
                interp = f"Democrats win {diff:.1f} more seats than the median neutral map."
            else:
                interp = f"Democrats win {abs(diff):.1f} fewer seats than the median neutral map."
        elif col == "mean_median":
            interp = (
                "Positive means Democrats are slightly over-represented in median district; "
                "negative means under-represented."
            )
        else:
            interp = ""

        comparison[col] = {
            "enacted_value": round(float(val), 4),
            "ensemble_median": round(float(s.median()), 4),
            "ensemble_p5":  round(float(s.quantile(0.05)), 4),
            "ensemble_p95": round(float(s.quantile(0.95)), 4),
            "percentile_rank": round(pct, 1),
            "is_outlier": pct < 5 or pct > 95,
            "interpretation": interp,
        }

    return {
        "chamber": chamber,
        "ensemble_size": len(df),
        "comparison": comparison,
    }


@app.get("/api/lrdb/jurisdiction/{name}")
def lrdb_jurisdiction(name: str):
    """
    Look up a local government jurisdiction from the LRDB dataset.
    Links fdga-chain back to the local redistricting database.
    """
    import geopandas as gpd

    lrdb_path = Path("../lrdb/public/assets/lrdb_web_20260216.geojson")
    if not lrdb_path.exists():
        raise HTTPException(status_code=503,
                            detail="LRDB data not found. Check LRDB_DATA_PATH.")

    gdf = gpd.read_file(str(lrdb_path))
    name_lower = name.lower()
    mask = gdf["name"].str.lower().str.contains(name_lower, na=False)
    results = gdf[mask][["id", "name", "dist_type", "pop20", "no_districts",
                          "status", "redist_complete"]].to_dict(orient="records")
    if not results:
        raise HTTPException(status_code=404, detail=f"No jurisdiction found matching '{name}'")
    return {"query": name, "results": results}


# ---------------------------------------------------------------------------
# Map endpoints
# ---------------------------------------------------------------------------

@app.get("/api/maps/{chamber}/enacted")
def enacted_map(chamber: Chamber):
    """Enacted district polygons as GeoJSON. Cached after first load."""
    cache_key = f"enacted_{chamber}"
    if cache_key not in _geo_cache:
        import geopandas as gpd
        shp = DISTRICT_SHAPEFILES.get(chamber)
        if not shp or not shp.exists():
            raise HTTPException(status_code=404, detail=f"District shapefile not found for {chamber}")
        gdf = gpd.read_file(str(shp)).to_crs("EPSG:4326")
        keep = [c for c in ["DISTRICT", "POPULATION", "NH_BLK", "HISPANIC_O", "geometry"] if c in gdf.columns]
        _geo_cache[cache_key] = json.loads(gdf[keep].to_json())
    return _geo_cache[cache_key]


@app.get("/api/maps/{chamber}/stability")
def stability_map(chamber: Chamber):
    """
    Precinct-level stability heatmap as GeoJSON.
    Each precinct has a 'stability' value (0–1): fraction of neutral maps
    where it was assigned to the same district as the enacted plan.
    High = always in same district. Low = often redrawn = contested boundary.
    """
    stability_path = ENSEMBLE_DIR / f"{chamber}_stability.json"
    if not stability_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No stability data for {chamber}. Re-run the ensemble to generate it."
        )
    stability = json.loads(stability_path.read_text())

    cache_key = f"precincts_simplified"
    if cache_key not in _geo_cache:
        import geopandas as gpd
        prec_path = RAW_DIR / "ga_precincts_ready.shp"
        if not prec_path.exists():
            raise HTTPException(status_code=404, detail="Precinct file not found. Run prep_data.py first.")
        gdf = gpd.read_file(str(prec_path)).to_crs("EPSG:4326")
        # Keep only geometry + district assignment columns; simplify for faster transfer
        keep_cols = [c for c in ["HDIST", "SDIST", "CDIST", "geometry"] if c in gdf.columns]
        gdf = gdf[keep_cols].copy()
        gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)
        _geo_cache[cache_key] = gdf
    gdf = _geo_cache[cache_key].copy()

    district_col = {"house": "HDIST", "senate": "SDIST", "congress": "CDIST"}[chamber]
    gdf["stability"] = [stability.get(str(i), None) for i in gdf.index]
    gdf["district"]  = gdf[district_col] if district_col in gdf.columns else None

    return json.loads(gdf[["geometry", "stability", "district"]].to_json())


# ---------------------------------------------------------------------------
# Ensemble run endpoints
# ---------------------------------------------------------------------------

# Tracks in-progress runs: run_id → {status, chamber, steps, ...}
_run_registry: dict = {}


def _load_run_registry():
    reg_path = ENSEMBLE_DIR / "runs.json"
    if reg_path.exists():
        return json.loads(reg_path.read_text())
    return {}


def _save_run_registry(registry: dict):
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    (ENSEMBLE_DIR / "runs.json").write_text(json.dumps(registry, indent=2))


@app.get("/api/ensemble/runs")
def list_runs():
    """List all ensemble runs (current + historical)."""
    registry = _load_run_registry()
    # Merge with in-process runs
    merged = {**registry, **_run_registry}
    return {"runs": list(merged.values())}


ALGO_INFO = {
    "recom": {
        "name": "ReCom (Recombination)",
        "speed": "fast",
        "description": "The standard ensemble method. Merges two adjacent districts and re-splits via a random spanning tree. Fast and widely used in academic redistricting studies.",
        "best_for": "General use, educational demos, quick exploration.",
    },
    "reversible_recom": {
        "name": "Reversible ReCom",
        "speed": "slower",
        "description": "ReCom with Metropolis-Hastings acceptance. Rejects some proposals to ensure the chain samples from a known probability distribution (detailed balance). Statistically more rigorous but ~20-30% slower.",
        "best_for": "Peer-reviewed analysis, reproducible research, when statistical rigor matters.",
    },
}


@app.get("/api/algorithms")
def list_algorithms():
    """Available proposal algorithms with descriptions."""
    return ALGO_INFO


@app.post("/api/ensemble/{chamber}/run")
def start_run(
    chamber: Chamber,
    background_tasks: BackgroundTasks,
    steps: int   = Query(default=10000, ge=100, le=100000),
    epsilon: float = Query(default=0.07, ge=0.01, le=0.15),
    seed: int | None = Query(default=None),
    algo: str = Query(default="recom"),
):
    """
    Trigger a new ensemble run in the background.
    Returns a run_id to poll for status.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if algo not in ALGO_INFO:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm '{algo}'. Options: {list(ALGO_INFO)}")
    run_info = {
        "run_id":    run_id,
        "chamber":   chamber,
        "steps":     steps,
        "epsilon":   epsilon,
        "seed":      seed,
        "algo":      algo,
        "status":    "queued",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "error":     None,
    }
    _run_registry[run_id] = run_info

    def _run():
        _run_registry[run_id]["status"] = "running"
        cmd = [
            sys.executable, "scripts/run_ensemble.py",
            "--chamber", chamber,
            "--steps", str(steps),
            "--epsilon", str(epsilon),
            "--algo", algo,
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                _run_registry[run_id]["status"] = "done"
            else:
                _run_registry[run_id]["status"] = "error"
                _run_registry[run_id]["error"]  = result.stderr[-2000:]
        except Exception as e:
            _run_registry[run_id]["status"] = "error"
            _run_registry[run_id]["error"]  = str(e)
        finally:
            _run_registry[run_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            # Persist to registry file
            registry = _load_run_registry()
            registry[run_id] = _run_registry[run_id]
            _save_run_registry(registry)
            # Invalidate geo cache so stability map reloads
            _geo_cache.pop(f"enacted_{chamber}", None)

    background_tasks.add_task(_run)
    return run_info


@app.get("/api/ensemble/{chamber}/run/{run_id}/status")
def run_status(chamber: Chamber, run_id: str):
    """Poll status of a background ensemble run."""
    # Check in-memory first, then registry file
    if run_id in _run_registry:
        return _run_registry[run_id]
    registry = _load_run_registry()
    if run_id in registry:
        return registry[run_id]
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


# ---------------------------------------------------------------------------
# Info endpoint
# ---------------------------------------------------------------------------

@app.get("/api/info")
def info():
    """Static info about data sources and algorithm for the Sources & Config tab."""
    chambers_info = []
    for chamber in ["house", "senate", "congress"]:
        meta_path = ENSEMBLE_DIR / f"{chamber}_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            chambers_info.append({
                "chamber":          chamber,
                "steps":            meta.get("steps"),
                "epsilon":          meta.get("epsilon"),
                "num_districts":    meta.get("num_districts"),
                "total_population": meta.get("total_population"),
                "ideal_population": meta.get("ideal_population"),
                "runtime_seconds":  meta.get("runtime_seconds"),
                "ran_at":           meta.get("ran_at"),
                "algorithm":        meta.get("algorithm"),
                "data_sources":     meta.get("data_sources"),
            })

    return {
        "project": "fdga-chain",
        "description": "GerryChain ensemble analysis of Georgia redistricting maps",
        "organization": "Fair Districts Georgia — fairdistrictsga.org",
        "chambers": chambers_info,
        "methodology": {
            "summary": (
                "We generate thousands of alternative redistricting maps using a Markov chain "
                "random walk (ReCom algorithm). Each map satisfies only legal constraints: "
                "equal population, contiguous districts. We then measure where the enacted "
                "map falls in the distribution of these neutral alternatives."
            ),
            "metric_definitions": {
                "efficiency_gap": (
                    "Difference in wasted votes (votes for losing candidates + excess winning votes) "
                    "between parties, expressed as a fraction of total votes."
                ),
                "mean_median": (
                    "Difference between a party's mean vote share and median vote share across districts. "
                    "A large gap suggests the party's voters are spread inefficiently."
                ),
                "dem_seats": "Number of districts where Democrats win a majority of votes.",
                "polsby_popper": (
                    "Ratio of district area to the area of a circle with the same perimeter. "
                    "1.0 = perfect circle; lower = more irregular shape."
                ),
                "stability": (
                    "For each precinct: the fraction of neutral maps where it is assigned to "
                    "the same district as in the enacted plan. Low stability = contested boundary area."
                ),
            },
            "references": [
                "Metric definitions: Stephanopoulos & McGhee (2015), Bernstein & Duchin (2017)",
                "GerryChain: MGGG Redistricting Lab, Tufts University",
                "Data: Redistricting Data Hub (redistrictingdatahub.org)",
            ],
        },
    }


# Static files — must be mounted AFTER all @app.get routes
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
