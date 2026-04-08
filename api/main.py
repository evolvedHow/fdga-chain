"""
fdga-chain API — state-agnostic redistricting ensemble analysis.

State-aware endpoints:
  GET /api/states                                      — list available states
  GET /api/states/{state}/info                         — state metadata + cycle list
  GET /api/states/{state}/{chamber}/cycles             — redistricting cycles
  GET /api/states/{state}/{chamber}/boundaries/{year}  — GeoJSON for a cycle
  GET /api/states/{state}/{chamber}/metrics/{year}     — efficiency gap, wasted votes, etc.
  GET /api/states/{state}/{chamber}/ensemble/summary   — GerryChain distribution stats
  GET /api/states/{state}/{chamber}/ensemble/histogram — histogram for one metric
  GET /api/states/{state}/{chamber}/ensemble/enacted   — enacted plan vs ensemble
  GET /api/maps/{state}/{chamber}/enacted              — GeoJSON (most recent enacted)
  GET /api/maps/{state}/{chamber}/stability            — precinct stability heatmap
  POST /api/states/{state}/{chamber}/ensemble/run      — trigger async GerryChain run
  GET /api/ensemble/runs                               — list all runs

Legacy (GA-only shortcuts, kept for backward compat during transition):
  GET /api/chambers                    → same as /api/states/GA/info chambers
  GET /api/ensemble/{chamber}/summary  → /api/states/GA/{chamber}/ensemble/summary
  GET /api/ensemble/{chamber}/histogram
  GET /api/ensemble/{chamber}/enacted
  GET /api/maps/{chamber}/enacted
  GET /api/maps/{chamber}/stability
  GET /api/info
  GET /health

Run:
  uv run uvicorn api.main:app --reload --port 8001
"""

import json
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()  # reads .env if present (no-op in Modal where env vars come from Secrets)
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.data_loader import (
    get_state_config,
    invalidate_ensemble_cache,
    invalidate_geo_cache,
    list_states,
    load_cycle_boundaries,
    load_cycle_elections,
    load_cycle_metrics,
    load_enacted_geojson,
    load_ensemble,
    load_precinct_geodataframe,
    load_stability_json,
    LEGACY_ENSEMBLE_DIR,
)

app = FastAPI(title="fdga-chain", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "https://evolvedhow.github.io",
        # Wildcard for local dev on any port
        "http://localhost:*",
    ],
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path("frontend")
ENSEMBLE_DIR = LEGACY_ENSEMBLE_DIR  # legacy path still used by run scripts

METRICS = [
    "dem_seats", "efficiency_gap", "mean_median",
    "polsby_popper_mean", "polsby_popper_min",
    "majority_minority_districts", "num_cut_edges",
]

_run_registry: dict = {}

ALGO_INFO = {
    "recom": {
        "name": "ReCom (Recombination)",
        "speed": "fast",
        "description": (
            "The standard ensemble method. Merges two adjacent districts and re-splits "
            "via a random spanning tree. Fast and widely used in academic redistricting studies."
        ),
        "best_for": "General use, educational demos, quick exploration.",
    },
    "reversible_recom": {
        "name": "Reversible ReCom",
        "speed": "slower",
        "description": (
            "ReCom with Metropolis-Hastings acceptance. Rejects some proposals to ensure "
            "the chain samples from a known probability distribution (detailed balance). "
            "Statistically more rigorous but ~20-30% slower."
        ),
        "best_for": "Peer-reviewed analysis, reproducible research, when statistical rigor matters.",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(series: pd.Series, value: float) -> float:
    return float((series <= value).mean() * 100)


def _hist(series: pd.Series, bins: int = 30) -> dict:
    counts, edges = np.histogram(series.dropna(), bins=bins)
    return {"counts": counts.tolist(), "bin_edges": edges.tolist()}


def _load_run_registry() -> dict:
    reg_path = ENSEMBLE_DIR / "runs.json"
    if reg_path.exists():
        return json.loads(reg_path.read_text())
    return {}


def _save_run_registry(registry: dict):
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    (ENSEMBLE_DIR / "runs.json").write_text(json.dumps(registry, indent=2))


def _enacted_comparison(state: str, chamber: str) -> dict:
    """Core logic for enacted-vs-ensemble comparison (used by both legacy and new routes)."""
    df, meta = load_ensemble(state, chamber)
    enacted = meta.get("enacted_metrics", {})

    comparison = {}
    for col in METRICS:
        if col not in df.columns or col not in enacted:
            continue
        s = df[col].dropna()
        val = enacted[col]
        pct = _percentile(s, val)

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
            "enacted_value":    round(float(val), 4),
            "ensemble_median":  round(float(s.median()), 4),
            "ensemble_p5":      round(float(s.quantile(0.05)), 4),
            "ensemble_p95":     round(float(s.quantile(0.95)), 4),
            "ensemble_p25":     round(float(s.quantile(0.25)), 4),
            "ensemble_p75":     round(float(s.quantile(0.75)), 4),
            "percentile_rank":  round(pct, 1),
            "is_outlier":       pct < 5 or pct > 95,
            "interpretation":   interp,
            "ensemble_size":    len(df),
        }
    return {"chamber": chamber, "state": state, "ensemble_size": len(df), "comparison": comparison}


# ---------------------------------------------------------------------------
# Root / health
# ---------------------------------------------------------------------------

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/config")
def api_config():
    """
    Public config for the frontend. Returns non-sensitive settings and
    client-facing tokens (Mapbox public key is intentionally public —
    restrict it by URL in your Mapbox dashboard, not by keeping it secret).
    Reads from environment variables set via .env or Modal Secrets.
    """
    return {
        "mapbox_token": os.environ.get("MAPBOX_TOKEN", ""),
        "active_state": os.environ.get("ACTIVE_STATE", "GA"),
    }


@app.get("/health")
def health():
    states = list_states()
    chambers_ready = []
    for state in states:
        for chamber in ["house", "senate", "congress"]:
            try:
                load_ensemble(state, chamber)
                chambers_ready.append(f"{state}/{chamber}")
            except FileNotFoundError:
                pass
    return {"status": "ok", "states": states, "chambers_ready": chambers_ready}


# ---------------------------------------------------------------------------
# State-aware routes
# ---------------------------------------------------------------------------

@app.get("/api/states")
def api_list_states():
    """List all states that have data available."""
    states = list_states()
    result = []
    for s in states:
        try:
            cfg = get_state_config(s)
            result.append({"state": s, "name": cfg.get("name", s)})
        except FileNotFoundError:
            result.append({"state": s, "name": s})
    return result


@app.get("/api/states/{state}/info")
def api_state_info(state: str):
    """State metadata: chambers, redistricting cycles, timeline events."""
    state = state.upper()
    try:
        cfg = get_state_config(state)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    return cfg


@app.get("/api/states/{state}/{chamber}/cycles")
def api_chamber_cycles(state: str, chamber: str):
    """List redistricting cycles available for a state/chamber."""
    state = state.upper()
    chamber = chamber.lower()
    try:
        cfg = get_state_config(state)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    cycles = cfg.get("cycles", [])
    # Annotate with data availability
    from api.data_loader import get_cycle_dir
    result = []
    for c in cycles:
        cycle_dir = get_cycle_dir(state, chamber, c["year"])
        result.append({
            **c,
            "has_boundaries": (cycle_dir / "boundaries.geojson").exists(),
            "has_elections":  (cycle_dir / "elections.json").exists(),
            "has_metrics":    (cycle_dir / "metrics.json").exists(),
        })
    return result


@app.get("/api/states/{state}/{chamber}/boundaries/{year}")
def api_boundaries(state: str, chamber: str, year: int):
    """GeoJSON boundaries for a state/chamber/cycle year."""
    state, chamber = state.upper(), chamber.lower()
    try:
        return load_cycle_boundaries(state, chamber, year)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/states/{state}/{chamber}/elections/{year}")
def api_elections(state: str, chamber: str, year: int):
    """District-level election results for a cycle year."""
    state, chamber = state.upper(), chamber.lower()
    try:
        return load_cycle_elections(state, chamber, year)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/states/{state}/{chamber}/metrics/{year}")
def api_cycle_metrics(state: str, chamber: str, year: int):
    """Precomputed redistricting metrics for a cycle (efficiency gap, wasted votes, seats/votes)."""
    state, chamber = state.upper(), chamber.lower()
    try:
        return load_cycle_metrics(state, chamber, year)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/states/{state}/{chamber}/ensemble/summary")
def api_ensemble_summary(state: str, chamber: str):
    """GerryChain ensemble distribution stats for all metrics."""
    state, chamber = state.upper(), chamber.lower()
    try:
        df, meta = load_ensemble(state, chamber)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

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
            entry["enacted_value"]      = round(float(enacted[col]), 4)
            entry["enacted_percentile"] = round(_percentile(s, enacted[col]), 1)
        summary[col] = entry

    return {
        "state": state,
        "chamber": chamber,
        "total_plans": len(df),
        "num_districts": meta.get("num_districts"),
        "epsilon": meta.get("epsilon"),
        "metrics": summary,
    }


@app.get("/api/states/{state}/{chamber}/ensemble/histogram")
def api_ensemble_histogram(
    state: str,
    chamber: str,
    metric: str = "dem_seats",
    bins: int = 30,
):
    """Histogram data for a single metric (for bar charts)."""
    state, chamber = state.upper(), chamber.lower()
    if metric not in METRICS:
        raise HTTPException(status_code=400, detail=f"Unknown metric. Choose from: {METRICS}")
    try:
        df, meta = load_ensemble(state, chamber)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if metric not in df.columns:
        raise HTTPException(status_code=404, detail=f"Metric '{metric}' not in ensemble data")

    enacted = meta.get("enacted_metrics", {})
    hist = _hist(df[metric], bins=bins)
    result = {
        "state": state,
        "chamber": chamber,
        "metric": metric,
        "histogram": hist,
        "total_plans": len(df),
    }
    if metric in enacted:
        result["enacted_value"]      = enacted[metric]
        result["enacted_percentile"] = round(_percentile(df[metric].dropna(), enacted[metric]), 1)
    return result


@app.get("/api/states/{state}/{chamber}/ensemble/enacted")
def api_enacted_comparison(state: str, chamber: str):
    """Enacted plan vs ensemble — the primary education endpoint."""
    state, chamber = state.upper(), chamber.lower()
    try:
        return _enacted_comparison(state, chamber)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/states/{state}/{chamber}/ensemble/run")
def api_start_run(
    state: str,
    chamber: str,
    background_tasks: BackgroundTasks,
    steps:   int   = Query(default=10000, ge=100, le=100000),
    epsilon: float = Query(default=0.07,  ge=0.01, le=0.15),
    seed:    int | None = Query(default=None),
    algo:    str  = Query(default="recom"),
):
    """Trigger a new GerryChain ensemble run in the background."""
    state, chamber = state.upper(), chamber.lower()
    if algo not in ALGO_INFO:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm. Options: {list(ALGO_INFO)}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_info = {
        "run_id":      run_id,
        "state":       state,
        "chamber":     chamber,
        "steps":       steps,
        "epsilon":     epsilon,
        "seed":        seed,
        "algo":        algo,
        "status":      "queued",
        "started_at":  datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "error":       None,
    }
    _run_registry[run_id] = run_info

    def _run():
        _run_registry[run_id]["status"] = "running"
        cmd = [
            sys.executable, "scripts/run_ensemble.py",
            "--chamber", chamber,
            "--steps",   str(steps),
            "--epsilon", str(epsilon),
            "--algo",    algo,
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
            registry = _load_run_registry()
            registry[run_id] = _run_registry[run_id]
            _save_run_registry(registry)
            invalidate_ensemble_cache(state, chamber)
            invalidate_geo_cache(f"enacted_geojson_{state}_{chamber}_2021")

    background_tasks.add_task(_run)
    return run_info


@app.get("/api/states/{state}/{chamber}/ensemble/run/{run_id}/status")
def api_run_status(state: str, chamber: str, run_id: str):
    if run_id in _run_registry:
        return _run_registry[run_id]
    registry = _load_run_registry()
    if run_id in registry:
        return registry[run_id]
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


# ---------------------------------------------------------------------------
# Map endpoints (state-aware)
# ---------------------------------------------------------------------------

@app.get("/api/maps/{state}/{chamber}/enacted")
def api_enacted_map(state: str, chamber: str, year: int | None = None):
    """Enacted district polygons as GeoJSON (most recent cycle by default)."""
    state, chamber = state.upper(), chamber.lower()
    try:
        return load_enacted_geojson(state, chamber, year)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/maps/{state}/{chamber}/stability")
def api_stability_map(state: str, chamber: str):
    """Precinct-level stability heatmap as GeoJSON."""
    state, chamber = state.upper(), chamber.lower()
    try:
        stability = load_stability_json(state, chamber)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        gdf = load_precinct_geodataframe(state).copy()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    col_map = {"house": "HDIST", "senate": "SDIST", "congress": "CDIST"}
    district_col = col_map.get(chamber)
    gdf["stability"] = [stability.get(str(i), None) for i in gdf.index]
    gdf["district"]  = gdf[district_col] if district_col and district_col in gdf.columns else None

    return json.loads(gdf[["geometry", "stability", "district"]].to_json())


# ---------------------------------------------------------------------------
# Legacy GA-only routes (backward compatibility)
# ---------------------------------------------------------------------------

@app.get("/api/chambers")
def legacy_chambers():
    """Legacy: list GA chambers with ensemble data."""
    results = []
    for chamber in ["house", "senate", "congress"]:
        try:
            _, meta = load_ensemble("GA", chamber)
            results.append({
                "chamber":          chamber,
                "steps":            meta.get("steps"),
                "num_districts":    meta.get("num_districts"),
                "runtime_seconds":  meta.get("runtime_seconds"),
            })
        except FileNotFoundError:
            pass
    return results


@app.get("/api/ensemble/{chamber}/summary")
def legacy_summary(chamber: str):
    return api_ensemble_summary("GA", chamber)


@app.get("/api/ensemble/{chamber}/histogram")
def legacy_histogram(chamber: str, metric: str = "dem_seats", bins: int = 30):
    return api_ensemble_histogram("GA", chamber, metric, bins)


@app.get("/api/ensemble/{chamber}/enacted")
def legacy_enacted(chamber: str):
    return api_enacted_comparison("GA", chamber)


@app.post("/api/ensemble/{chamber}/run")
def legacy_run(
    chamber: str,
    background_tasks: BackgroundTasks,
    steps:   int   = Query(default=10000, ge=100, le=100000),
    epsilon: float = Query(default=0.07,  ge=0.01, le=0.15),
    seed:    int | None = Query(default=None),
    algo:    str  = Query(default="recom"),
):
    return api_start_run("GA", chamber, background_tasks, steps, epsilon, seed, algo)


@app.get("/api/ensemble/{chamber}/run/{run_id}/status")
def legacy_run_status(chamber: str, run_id: str):
    return api_run_status("GA", chamber, run_id)


@app.get("/api/ensemble/runs")
def legacy_list_runs():
    registry = _load_run_registry()
    merged = {**registry, **_run_registry}
    return {"runs": list(merged.values())}


@app.get("/api/maps/{chamber}/enacted")
def legacy_enacted_map(chamber: str):
    return api_enacted_map("GA", chamber)


@app.get("/api/maps/{chamber}/stability")
def legacy_stability_map(chamber: str):
    return api_stability_map("GA", chamber)


@app.get("/api/algorithms")
def list_algorithms():
    return ALGO_INFO


@app.get("/api/info")
def info():
    chambers_info = []
    for chamber in ["house", "senate", "congress"]:
        try:
            _, meta = load_ensemble("GA", chamber)
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
        except FileNotFoundError:
            pass

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
                    "Difference in wasted votes between parties, expressed as a fraction of total votes."
                ),
                "mean_median": (
                    "Difference between a party's mean and median vote share across districts."
                ),
                "dem_seats": "Number of districts where Democrats win a majority of votes.",
                "polsby_popper": (
                    "Ratio of district area to the area of a circle with the same perimeter. "
                    "1.0 = perfect circle."
                ),
                "stability": (
                    "For each precinct: fraction of neutral maps where it stays in the same district. "
                    "Low = contested boundary area."
                ),
            },
            "references": [
                "Stephanopoulos & McGhee (2015), Bernstein & Duchin (2017)",
                "GerryChain: MGGG Redistricting Lab, Tufts University",
                "Data: Redistricting Data Hub (redistrictingdatahub.org)",
            ],
        },
    }


@app.get("/api/lrdb/jurisdiction/{name}")
def lrdb_jurisdiction(name: str):
    import geopandas as gpd
    lrdb_path = Path("../lrdb/public/assets/lrdb_web_20260216.geojson")
    if not lrdb_path.exists():
        raise HTTPException(status_code=503, detail="LRDB data not found.")
    gdf = gpd.read_file(str(lrdb_path))
    mask = gdf["name"].str.lower().str.contains(name.lower(), na=False)
    results = gdf[mask][["id", "name", "dist_type", "pop20", "no_districts",
                          "status", "redist_complete"]].to_dict(orient="records")
    if not results:
        raise HTTPException(status_code=404, detail=f"No jurisdiction found matching '{name}'")
    return {"query": name, "results": results}


# ---------------------------------------------------------------------------
# Static file serving — must be LAST
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
