"""
fetch_boundaries.py — Download and convert historical redistricting boundary files to GeoJSON.

Data sources:
  - U.S. Census TIGER/Line Shapefiles (congressional districts, 106th–118th Congress)
  - Census State Legislative Districts (upper/lower) via NHGIS or direct TIGER download
  - Redistricting Data Hub for state-specific enacted shapefiles

Output layout:
  data/states/{STATE}/{chamber}/{year}/boundaries.geojson

Usage:
  uv run python scripts/fetch_boundaries.py --state GA
  uv run python scripts/fetch_boundaries.py --state GA --chamber congress --year 2011
  uv run python scripts/fetch_boundaries.py --state GA --force   # re-download even if file exists
"""

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd

# ---------------------------------------------------------------------------
# Catalog of available boundary sources
# ---------------------------------------------------------------------------

# TIGER/Line congressional district shapefiles by Congress number
# Format: https://www2.census.gov/geo/tiger/TIGER{year}/CD/tl_{year}_{fips}_cd{congress}.zip
CONGRESS_TIGER = {
    # 107th Congress enacted after 2000 Census (used 2002–2012)
    2001: {"congress_num": "107", "tiger_year": "2010"},
    # 113th Congress enacted after 2010 Census (used 2012–2022)
    2011: {"congress_num": "113", "tiger_year": "2014"},
    # 118th Congress enacted after 2020 Census
    2021: {"congress_num": "118", "tiger_year": "2022"},
}

# TIGER/Line state legislative district shapefiles
# Format: https://www2.census.gov/geo/tiger/TIGER{year}/SLDL/ (lower) or SLDU/ (upper)
STATE_LEG_TIGER = {
    2001: {"tiger_year": "2010"},   # use 2010 TIGER (earliest widely available)
    2005: {"tiger_year": "2010"},   # GA mid-decade re-draw; use 2010 TIGER as proxy
    2011: {"tiger_year": "2012"},
    2021: {"tiger_year": "2022"},
}

# State FIPS codes
STATE_FIPS = {
    "GA": "13",
    "NC": "37",
    "TX": "48",
    "FL": "12",
    "PA": "42",
    "WI": "55",
    "MI": "26",
    "OH": "39",
    "VA": "51",
    "IL": "17",
}

# Chamber → TIGER directory and layer suffix
CHAMBER_TIGER = {
    "congress": {"dir": "CD",   "prefix": "cd",  "type": "congress"},
    "senate":   {"dir": "SLDU", "prefix": "sldu", "type": "upper"},
    "house":    {"dir": "SLDL", "prefix": "sldl", "type": "lower"},
}

BASE_TIGER_URL = "https://www2.census.gov/geo/tiger"


def tiger_url_congress(tiger_year: str, fips: str, congress_num: str) -> str:
    return (
        f"{BASE_TIGER_URL}/TIGER{tiger_year}/CD/"
        f"tl_{tiger_year}_{fips}_cd{congress_num}.zip"
    )


def tiger_url_state_leg(tiger_year: str, fips: str, chamber: str) -> str:
    info = CHAMBER_TIGER[chamber]
    tiger_dir = info["dir"]
    prefix = info["prefix"]
    return (
        f"{BASE_TIGER_URL}/TIGER{tiger_year}/{tiger_dir}/"
        f"tl_{tiger_year}_{fips}_{prefix}.zip"
    )


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_zip(url: str, dest_dir: Path) -> Path:
    """Download a zip file, extract to dest_dir, return dest_dir."""
    print(f"  ↓ {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
    except Exception as e:
        raise RuntimeError(f"Download failed: {url}\n  {e}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dest_dir)
    return dest_dir


def find_shapefile(directory: Path) -> Path | None:
    shps = list(directory.glob("*.shp"))
    return shps[0] if shps else None


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def fetch_boundary(state: str, chamber: str, year: int, force: bool = False) -> Path:
    """
    Download and convert one boundary shapefile to GeoJSON.
    Returns the output GeoJSON path.
    """
    fips = STATE_FIPS.get(state.upper())
    if not fips:
        raise ValueError(f"Unknown state '{state}'. Add to STATE_FIPS dict.")

    out_dir = Path(f"data/states/{state.upper()}/{chamber.lower()}/{year}")
    out_path = out_dir / "boundaries.geojson"

    if out_path.exists() and not force:
        print(f"  ✓ Already exists: {out_path} (use --force to re-download)")
        return out_path

    # Determine download URL
    tmp_dir = Path(f"/tmp/fdga_boundaries/{state}_{chamber}_{year}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if chamber == "congress":
        cfg = CONGRESS_TIGER.get(year)
        if not cfg:
            raise ValueError(f"No congressional boundary config for year {year}. "
                             f"Available: {list(CONGRESS_TIGER)}")
        url = tiger_url_congress(cfg["tiger_year"], fips, cfg["congress_num"])
    else:
        cfg = STATE_LEG_TIGER.get(year)
        if not cfg:
            raise ValueError(f"No state legislative boundary config for year {year}. "
                             f"Available: {list(STATE_LEG_TIGER)}")
        url = tiger_url_state_leg(cfg["tiger_year"], fips, chamber)

    # Download + extract
    try:
        download_zip(url, tmp_dir)
    except RuntimeError as e:
        print(f"  WARNING: {e}")
        print("  You may need to download this file manually from:")
        print(f"    https://redistrictingdatahub.org/state/{state.lower()}/")
        print(f"    or https://www2.census.gov/geo/tiger/")
        return None

    shp = find_shapefile(tmp_dir)
    if not shp:
        print(f"  ERROR: No shapefile found in {tmp_dir}")
        return None

    # Load, reproject, simplify, convert
    print(f"  Converting {shp.name} → GeoJSON…")
    gdf = gpd.read_file(str(shp)).to_crs("EPSG:4326")

    # Filter to this state (TIGER files sometimes contain all states)
    if "STATEFP" in gdf.columns:
        gdf = gdf[gdf["STATEFP"] == fips].copy()
    elif "GEOID" in gdf.columns and len(fips) == 2:
        gdf = gdf[gdf["GEOID"].str.startswith(fips)].copy()

    if len(gdf) == 0:
        print(f"  ERROR: No features for FIPS {fips} in shapefile")
        return None

    # Normalize district ID column
    district_col = None
    for col in ["CD118FP", "CD113FP", "CD107FP", "SLDUST", "SLDLST",
                "NAME", "NAMELSAD", "GEOID"]:
        if col in gdf.columns:
            district_col = col
            break

    # Simplify geometry slightly for faster transfer (tolerance ~50m in degrees)
    gdf["geometry"] = gdf.geometry.simplify(0.0005, preserve_topology=True)

    # Keep only useful columns
    keep = ["geometry"]
    for col in [district_col, "GEOID", "NAMELSAD", "ALAND", "AWATER"]:
        if col and col in gdf.columns:
            keep.append(col)
    gdf = gdf[list(dict.fromkeys(keep))].copy()  # deduplicate

    # Add a normalized 'district_id' field
    if district_col and district_col in gdf.columns:
        gdf["district_id"] = gdf[district_col].astype(str).str.lstrip("0").str.strip()

    out_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out_path), driver="GeoJSON")
    size_kb = out_path.stat().st_size / 1024
    print(f"  ✓ {out_path}  ({len(gdf)} districts, {size_kb:.0f} kB)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical redistricting boundary files and convert to GeoJSON."
    )
    parser.add_argument("--state",   required=True, help="State abbreviation (e.g. GA)")
    parser.add_argument("--chamber", default=None,
                        choices=["house", "senate", "congress"],
                        help="Specific chamber (default: all)")
    parser.add_argument("--year",    type=int, default=None,
                        help="Specific year (default: all configured years)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-download even if GeoJSON already exists")
    args = parser.parse_args()

    state = args.state.upper()
    chambers = [args.chamber] if args.chamber else ["house", "senate", "congress"]

    all_years = sorted(set(list(CONGRESS_TIGER.keys()) + list(STATE_LEG_TIGER.keys())))
    years = [args.year] if args.year else all_years

    print(f"\nFetching boundaries for {state}")
    print(f"Chambers: {chambers}")
    print(f"Years: {years}\n")

    errors = []
    for chamber in chambers:
        for year in years:
            print(f"── {chamber} {year} ──")
            try:
                result = fetch_boundary(state, chamber, year, force=args.force)
                if result is None:
                    errors.append(f"{chamber}/{year}")
            except Exception as e:
                print(f"  ERROR: {e}")
                errors.append(f"{chamber}/{year}")
            print()

    if errors:
        print(f"Failed or skipped: {errors}")
        print("\nFor missing years, download manually from:")
        print("  https://redistrictingdatahub.org/state/" + state.lower() + "/")
        print("  Place shapefiles at: data/states/{state}/{chamber}/{year}/")
        print("  Then re-run this script to convert.")
    else:
        print("Done. All boundaries downloaded successfully.")
    print("\nNext: run scripts/fetch_elections.py --state " + state)


if __name__ == "__main__":
    main()
