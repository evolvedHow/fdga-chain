"""
data_loader.py — State-aware data loading with in-memory caching.

Data layout on disk:
  data/
    states/
      GA/
        config.json                     ← state metadata, cycles, chambers
        house/
          2001/ boundaries.geojson, elections.json, metrics.json
          2011/ ...
          2021/ ...
          ensemble.parquet              ← GerryChain run (current/most recent)
          ensemble_meta.json
          stability.json                ← precinct stability scores
        senate/ ...
        congress/ ...
      NC/ ...                           ← adding a state = drop files here

Legacy fallback (GA only):
  data/ensembles/{chamber}_ensemble.parquet
  data/ensembles/{chamber}_meta.json
  data/ensembles/{chamber}_stability.json
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

# Root data directory (relative to where uvicorn is started, i.e. project root)
DATA_ROOT = Path("data")
STATES_DIR = DATA_ROOT / "states"
LEGACY_ENSEMBLE_DIR = DATA_ROOT / "ensembles"

# Geo cache: avoid re-loading large GeoJSON/parquet on every request
_geo_cache: dict[str, Any] = {}
_ensemble_cache: dict[str, tuple[pd.DataFrame, dict]] = {}


# ---------------------------------------------------------------------------
# State discovery
# ---------------------------------------------------------------------------

def list_states() -> list[str]:
    """Return sorted list of states that have a config.json."""
    if not STATES_DIR.exists():
        return []
    return sorted(
        d.name for d in STATES_DIR.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    )


def get_state_config(state: str) -> dict:
    """Load and return config.json for a state. Raises FileNotFoundError if missing."""
    path = STATES_DIR / state.upper() / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"No config found for state '{state}'. Expected: {path}")
    return json.loads(path.read_text())


def get_state_dir(state: str) -> Path:
    return STATES_DIR / state.upper()


def get_chamber_dir(state: str, chamber: str) -> Path:
    return get_state_dir(state) / chamber.lower()


def get_cycle_dir(state: str, chamber: str, year: int) -> Path:
    return get_chamber_dir(state, chamber) / str(year)


# ---------------------------------------------------------------------------
# Ensemble data
# ---------------------------------------------------------------------------

def load_ensemble(state: str, chamber: str) -> tuple[pd.DataFrame, dict]:
    """
    Load ensemble parquet + meta for a state/chamber.
    Checks new layout first, falls back to legacy data/ensembles/ for GA.
    Returns (DataFrame, meta_dict). Raises FileNotFoundError if not found.
    Caches result in memory.
    """
    cache_key = f"{state.upper()}_{chamber.lower()}"
    if cache_key in _ensemble_cache:
        return _ensemble_cache[cache_key]

    # New layout
    chamber_dir = get_chamber_dir(state, chamber)
    parquet_new = chamber_dir / "ensemble.parquet"
    meta_new = chamber_dir / "ensemble_meta.json"

    # Legacy layout (GA only)
    parquet_legacy = LEGACY_ENSEMBLE_DIR / f"{chamber.lower()}_ensemble.parquet"
    meta_legacy = LEGACY_ENSEMBLE_DIR / f"{chamber.lower()}_meta.json"

    if parquet_new.exists():
        parquet_path, meta_path = parquet_new, meta_new
    elif state.upper() == "GA" and parquet_legacy.exists():
        parquet_path, meta_path = parquet_legacy, meta_legacy
    else:
        raise FileNotFoundError(
            f"No ensemble data for {state}/{chamber}. "
            f"Run: uv run python scripts/run_ensemble.py --state {state} --chamber {chamber}"
        )

    df = pd.read_parquet(parquet_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    _ensemble_cache[cache_key] = (df, meta)
    return df, meta


def invalidate_ensemble_cache(state: str, chamber: str):
    """Call after a new ensemble run completes."""
    cache_key = f"{state.upper()}_{chamber.lower()}"
    _ensemble_cache.pop(cache_key, None)


# ---------------------------------------------------------------------------
# Historical cycle data (boundaries / elections / metrics)
# ---------------------------------------------------------------------------

def load_cycle_boundaries(state: str, chamber: str, year: int) -> dict:
    """Load boundaries.geojson for a given redistricting cycle. Returns parsed GeoJSON dict."""
    cache_key = f"boundaries_{state}_{chamber}_{year}"
    if cache_key in _geo_cache:
        return _geo_cache[cache_key]

    path = get_cycle_dir(state, chamber, year) / "boundaries.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"No boundary data for {state}/{chamber}/{year}. "
            f"Run: uv run python scripts/fetch_boundaries.py --state {state}"
        )
    data = json.loads(path.read_text())
    _geo_cache[cache_key] = data
    return data


def load_cycle_elections(state: str, chamber: str, year: int) -> dict:
    """Load elections.json for a cycle. Returns dict with district-level vote data."""
    path = get_cycle_dir(state, chamber, year) / "elections.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No election data for {state}/{chamber}/{year}. "
            f"Run: uv run python scripts/fetch_elections.py --state {state}"
        )
    return json.loads(path.read_text())


def load_cycle_metrics(state: str, chamber: str, year: int) -> dict:
    """Load precomputed metrics.json for a cycle (efficiency gap, wasted votes, etc.)."""
    path = get_cycle_dir(state, chamber, year) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No metrics for {state}/{chamber}/{year}. "
            f"Run: uv run python scripts/compute_metrics.py --state {state}"
        )
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Map / GeoJSON helpers
# ---------------------------------------------------------------------------

def load_enacted_geojson(state: str, chamber: str, year: int | None = None) -> dict:
    """
    Load enacted district GeoJSON for a state/chamber/year.
    If year is None, loads the most recent available cycle or falls back to raw shapefile.
    Caches result.
    """
    # Determine year to load
    if year is None:
        cfg = get_state_config(state)
        years = [c["year"] for c in cfg.get("cycles", [])]
        year = max(years) if years else 2021

    cache_key = f"enacted_geojson_{state}_{chamber}_{year}"
    if cache_key in _geo_cache:
        return _geo_cache[cache_key]

    # Try cycle directory first
    cycle_path = get_cycle_dir(state, chamber, year) / "boundaries.geojson"
    if cycle_path.exists():
        data = json.loads(cycle_path.read_text())
        _geo_cache[cache_key] = data
        return data

    # Legacy fallback: load from raw shapefile (GA only)
    if state.upper() == "GA":
        data = _load_legacy_shapefile(chamber)
        _geo_cache[cache_key] = data
        return data

    raise FileNotFoundError(
        f"No enacted GeoJSON for {state}/{chamber}/{year}. "
        f"Run: uv run python scripts/fetch_boundaries.py --state {state}"
    )


def _load_legacy_shapefile(chamber: str) -> dict:
    """Load GA enacted map from raw shapefiles (legacy path)."""
    import geopandas as gpd

    RAW_DIR = DATA_ROOT / "raw"
    shapefiles = {
        "house":    RAW_DIR / "House-2023 shape.shp",
        "senate":   RAW_DIR / "Senate-2023 shape file.shp",
        "congress": RAW_DIR / "Congress-2023 shape.shp",
    }
    shp = shapefiles.get(chamber)
    if not shp or not shp.exists():
        raise FileNotFoundError(f"District shapefile not found for {chamber}: {shp}")
    gdf = gpd.read_file(str(shp)).to_crs("EPSG:4326")
    keep = [c for c in ["DISTRICT", "POPULATION", "NH_BLK", "HISPANIC_O", "geometry"]
            if c in gdf.columns]
    return json.loads(gdf[keep].to_json())


def load_stability_json(state: str, chamber: str) -> dict:
    """Load precinct-level stability scores. Checks new layout then legacy."""
    # New layout
    new_path = get_chamber_dir(state, chamber) / "stability.json"
    if new_path.exists():
        return json.loads(new_path.read_text())

    # Legacy
    if state.upper() == "GA":
        legacy = LEGACY_ENSEMBLE_DIR / f"{chamber.lower()}_stability.json"
        if legacy.exists():
            return json.loads(legacy.read_text())

    raise FileNotFoundError(
        f"No stability data for {state}/{chamber}. "
        f"Re-run the ensemble to generate it."
    )


def load_precinct_geodataframe(state: str):
    """Load precinct GeoDataFrame for a state (cached). Returns geopandas GeoDataFrame."""
    cache_key = f"precincts_{state}"
    if cache_key in _geo_cache:
        return _geo_cache[cache_key]

    import geopandas as gpd

    # New layout: data/states/{state}/precincts.shp or .geojson
    state_dir = get_state_dir(state)
    for fname in ["precincts.geojson", "precincts.shp"]:
        p = state_dir / fname
        if p.exists():
            gdf = gpd.read_file(str(p)).to_crs("EPSG:4326")
            keep = [c for c in ["HDIST", "SDIST", "CDIST", "geometry"] if c in gdf.columns]
            gdf = gdf[keep].copy()
            gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)
            _geo_cache[cache_key] = gdf
            return gdf

    # Legacy fallback (GA)
    if state.upper() == "GA":
        prec_path = DATA_ROOT / "raw" / "ga_precincts_ready.shp"
        if prec_path.exists():
            gdf = gpd.read_file(str(prec_path)).to_crs("EPSG:4326")
            keep = [c for c in ["HDIST", "SDIST", "CDIST", "geometry"] if c in gdf.columns]
            gdf = gdf[keep].copy()
            gdf["geometry"] = gdf.geometry.simplify(0.001, preserve_topology=True)
            _geo_cache[cache_key] = gdf
            return gdf

    raise FileNotFoundError(f"No precinct file found for state '{state}'.")


def invalidate_geo_cache(key: str | None = None):
    """Clear geo cache entirely or for a specific key."""
    if key:
        _geo_cache.pop(key, None)
    else:
        _geo_cache.clear()
