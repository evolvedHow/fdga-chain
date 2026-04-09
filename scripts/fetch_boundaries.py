"""
fetch_boundaries.py — Download and convert historical redistricting boundary files to GeoJSON.

─── Manual download guide ────────────────────────────────────────────────────

If the auto-download fails, place files here and re-run — the script will
convert them without re-downloading:

  data/raw/manual/
    GA_congress_2001/     ← unzipped shapefile folder (any name, just needs a .shp inside)
    GA_congress_2011/
    GA_congress_2021/
    GA_house_2001/
    GA_house_2005/
    GA_house_2011/
    GA_house_2021/
    GA_senate_2001/
    GA_senate_2005/
    GA_senate_2011/
    GA_senate_2021/

  OR place the zip file directly:
    data/raw/manual/GA_congress_2001.zip
    data/raw/manual/GA_house_2011.zip
    etc.

─── Where to download ────────────────────────────────────────────────────────

Best source — Redistricting Data Hub (pre-clipped, pre-labeled):
  https://redistrictingdatahub.org/state/georgia/
  → "District" section → choose chamber and year → Download Shapefile

Second source — Census TIGER/Line (all-state files, larger):
  Congressional:
    https://www2.census.gov/geo/tiger/TIGER2022/CD/tl_2022_13_cd118.zip      (2021 maps, GA)
    https://www2.census.gov/geo/tiger/TIGER2013/CD/tl_2013_us_cd113.zip      (2011 maps, national — filtered to GA)
    https://www2.census.gov/geo/tiger/TIGER2010/CD/111/tl_2010_13_cd111.zip  (2001 maps, GA)

  State House (SLDL):
    https://www2.census.gov/geo/tiger/TIGER2022/SLDL/tl_2022_13_sldl.zip            (2021)
    https://www2.census.gov/geo/tiger/TIGER2012/SLDL/tl_2012_13_sldl.zip            (2011)
    https://www2.census.gov/geo/tiger/TIGER2010/SLDL/2010/tl_2010_13_sldl10.zip     (2001/2005)

  State Senate (SLDU):
    https://www2.census.gov/geo/tiger/TIGER2022/SLDU/tl_2022_13_sldu.zip            (2021)
    https://www2.census.gov/geo/tiger/TIGER2012/SLDU/tl_2012_13_sldu.zip            (2011)
    https://www2.census.gov/geo/tiger/TIGER2010/SLDU/2010/tl_2010_13_sldu10.zip     (2001/2005)

Third source — Dave's Redistricting (best for 2021 enacted maps):
  https://davesredistricting.org/maps#state::GA
  → click any map → "Export" → Shapefile

─── Output layout ────────────────────────────────────────────────────────────

  data/states/GA/congress/2021/boundaries.geojson
  data/states/GA/house/2021/boundaries.geojson
  data/states/GA/senate/2021/boundaries.geojson
  ... (same for 2001, 2005, 2011)

─── Usage ────────────────────────────────────────────────────────────────────

  # Auto-download all:
  uv run python scripts/fetch_boundaries.py --state GA

  # Convert only (if manual files already placed):
  uv run python scripts/fetch_boundaries.py --state GA --no-download

  # Single chamber/year:
  uv run python scripts/fetch_boundaries.py --state GA --chamber congress --year 2021

  # Re-convert even if GeoJSON already exists:
  uv run python scripts/fetch_boundaries.py --state GA --force
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
# Source catalog
# ---------------------------------------------------------------------------

# GA FIPS = 13. TIGER URLs below are GA-specific where possible.
TIGER_URLS = {
    # Congressional — GA-specific files (FIPS 13); 2011 uses national file filtered to GA
    # Note: 2005 mid-decade redraw was state legislature only; congress used 2001 map through 2010
    ("GA", "congress", 2001): "https://www2.census.gov/geo/tiger/TIGER2010/CD/111/tl_2010_13_cd111.zip",
    ("GA", "congress", 2005): "https://www2.census.gov/geo/tiger/TIGER2010/CD/111/tl_2010_13_cd111.zip",
    ("GA", "congress", 2011): "https://www2.census.gov/geo/tiger/TIGER2013/CD/tl_2013_us_cd113.zip",
    ("GA", "congress", 2021): "https://www2.census.gov/geo/tiger/TIGER2022/CD/tl_2022_13_cd118.zip",
    # State House (SLDL) — GA-specific files
    ("GA", "house", 2001): "https://www2.census.gov/geo/tiger/TIGER2010/SLDL/2010/tl_2010_13_sldl10.zip",
    ("GA", "house", 2005): "https://www2.census.gov/geo/tiger/TIGER2010/SLDL/2010/tl_2010_13_sldl10.zip",
    ("GA", "house", 2011): "https://www2.census.gov/geo/tiger/TIGER2012/SLDL/tl_2012_13_sldl.zip",
    ("GA", "house", 2021): "https://www2.census.gov/geo/tiger/TIGER2022/SLDL/tl_2022_13_sldl.zip",
    # State Senate (SLDU) — GA-specific files
    ("GA", "senate", 2001): "https://www2.census.gov/geo/tiger/TIGER2010/SLDU/2010/tl_2010_13_sldu10.zip",
    ("GA", "senate", 2005): "https://www2.census.gov/geo/tiger/TIGER2010/SLDU/2010/tl_2010_13_sldu10.zip",
    ("GA", "senate", 2011): "https://www2.census.gov/geo/tiger/TIGER2012/SLDU/tl_2012_13_sldu.zip",
    ("GA", "senate", 2021): "https://www2.census.gov/geo/tiger/TIGER2022/SLDU/tl_2022_13_sldu.zip",
}

STATE_FIPS = {"GA": "13", "NC": "37", "TX": "48", "FL": "12", "PA": "42",
              "WI": "55", "MI": "26", "OH": "39", "VA": "51", "IL": "17"}

MANUAL_DIR = Path("data/raw/manual")

# ---------------------------------------------------------------------------
# Manual file finder
# ---------------------------------------------------------------------------

def find_manual_source(state: str, chamber: str, year: int) -> Path | None:
    """
    Look for a manually placed file matching this state/chamber/year.
    Checks for:
      - data/raw/manual/{STATE}_{chamber}_{year}.zip
      - data/raw/manual/{STATE}_{chamber}_{year}/  (any .shp inside)
      - data/raw/manual/{STATE}_{chamber}_{year}*.zip  (glob)
    """
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)

    # Exact zip
    for pattern in [
        f"{state}_{chamber}_{year}.zip",
        f"{state.lower()}_{chamber}_{year}.zip",
        f"{chamber}_{year}.zip",
        f"*{chamber}*{year}*.zip",
        f"*{year}*{chamber}*.zip",
    ]:
        matches = list(MANUAL_DIR.glob(pattern))
        if matches:
            return matches[0]

    # Unzipped folder
    for pattern in [
        f"{state}_{chamber}_{year}",
        f"{state.lower()}_{chamber}_{year}",
        f"{chamber}_{year}",
    ]:
        folder = MANUAL_DIR / pattern
        if folder.is_dir() and list(folder.glob("*.shp")):
            return folder

    # Any folder containing matching keywords
    for folder in MANUAL_DIR.iterdir():
        if folder.is_dir() and str(year) in folder.name and chamber in folder.name.lower():
            if list(folder.glob("*.shp")):
                return folder

    return None


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_zip(url: str, dest_dir: Path) -> Path:
    print(f"    ↓ {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fdga-chain/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dest_dir)
    return dest_dir


def find_shapefile(directory: Path) -> Path | None:
    shps = list(directory.rglob("*.shp"))
    return shps[0] if shps else None


def extract_zip_to_tmp(zip_path: Path, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_dir)
    return tmp_dir


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_to_geojson(shp_path: Path, state: str, out_path: Path) -> bool:
    fips = STATE_FIPS.get(state.upper(), "")
    print(f"    Converting {shp_path.name}…")
    gdf = gpd.read_file(str(shp_path)).to_crs("EPSG:4326")

    # Filter to this state if the file covers multiple states
    if "STATEFP" in gdf.columns and fips:
        before = len(gdf)
        gdf = gdf[gdf["STATEFP"] == fips].copy()
        if len(gdf) == 0:
            print(f"    WARNING: No features for FIPS {fips} — trying without filter")
            gdf = gpd.read_file(str(shp_path)).to_crs("EPSG:4326")
        elif len(gdf) < before:
            print(f"    Filtered {before} → {len(gdf)} features for {state}")

    if len(gdf) == 0:
        print("    ERROR: Empty GeoDataFrame — check shapefile content")
        return False

    # Normalize a district_id column from whatever the source provides
    id_candidates = ["CD118FP", "CD113FP", "CD111FP", "SLDUST", "SLDLST",
                     "DISTRICT", "DIST_NUM", "NAME", "NAMELSAD", "GEOID"]
    district_col = next((c for c in id_candidates if c in gdf.columns), None)

    gdf["geometry"] = gdf.geometry.simplify(0.0005, preserve_topology=True)

    keep = ["geometry"]
    for col in [district_col, "GEOID", "NAMELSAD", "ALAND", "AWATER", "POPULATION"]:
        if col and col in gdf.columns and col not in keep:
            keep.append(col)
    gdf = gdf[keep].copy()

    if district_col and district_col in gdf.columns:
        gdf["district_id"] = gdf[district_col].astype(str).str.lstrip("0").str.strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out_path), driver="GeoJSON")
    size_kb = out_path.stat().st_size / 1024
    try:
        rel = out_path.relative_to(Path.cwd())
    except ValueError:
        rel = out_path
    print(f"    ✓ {rel}  ({len(gdf)} districts, {size_kb:.0f} kB)")
    return True


# ---------------------------------------------------------------------------
# Main entry point per chamber/year
# ---------------------------------------------------------------------------

def fetch_boundary(state: str, chamber: str, year: int,
                   force: bool = False, no_download: bool = False) -> bool:
    out_path = Path(f"data/states/{state.upper()}/{chamber.lower()}/{year}/boundaries.geojson")

    if out_path.exists() and not force:
        print(f"    ✓ Already exists (use --force to re-convert)")
        return True

    # 1. Check for manually placed files first
    manual = find_manual_source(state, chamber, year)
    if manual:
        print(f"    Found manual source: {manual}")
        if manual.suffix == ".zip":
            tmp = Path(f"/tmp/fdga_manual/{state}_{chamber}_{year}")
            extract_zip_to_tmp(manual, tmp)
            shp = find_shapefile(tmp)
        else:
            shp = find_shapefile(manual)

        if not shp:
            print(f"    ERROR: No .shp found in {manual}")
            return False
        return convert_to_geojson(shp, state, out_path)

    # 2. Auto-download
    if no_download:
        print(f"    SKIP — no manual file found and --no-download set")
        print(f"    Place a shapefile in: {MANUAL_DIR}/{state}_{chamber}_{year}/")
        return False

    url = TIGER_URLS.get((state.upper(), chamber.lower(), year))
    if not url:
        print(f"    No download URL configured for {state}/{chamber}/{year}")
        _print_manual_instructions(state, chamber, year)
        return False

    tmp = Path(f"/tmp/fdga_boundaries/{state}_{chamber}_{year}")
    try:
        download_zip(url, tmp)
    except RuntimeError as e:
        print(f"    Download failed: {e}")
        _print_manual_instructions(state, chamber, year)
        return False

    shp = find_shapefile(tmp)
    if not shp:
        print(f"    ERROR: No .shp found after download")
        _print_manual_instructions(state, chamber, year)
        return False

    return convert_to_geojson(shp, state, out_path)


def _print_manual_instructions(state: str, chamber: str, year: int):
    print()
    print(f"    ── Manual download for {state}/{chamber}/{year} ──")
    print(f"    1. Download a shapefile from one of:")
    print(f"       • https://redistrictingdatahub.org/state/{state.lower()}/")
    if (state.upper(), chamber, year) in TIGER_URLS:
        print(f"       • {TIGER_URLS[(state.upper(), chamber, year)]}")
    print(f"    2. Unzip into:  {MANUAL_DIR}/{state}_{chamber}_{year}/")
    print(f"       OR place zip at: {MANUAL_DIR}/{state}_{chamber}_{year}.zip")
    print(f"    3. Re-run: python scripts/fetch_boundaries.py --state {state} --chamber {chamber} --year {year}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch or convert redistricting boundary shapefiles to GeoJSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--state",       required=True, help="State abbreviation (e.g. GA)")
    parser.add_argument("--chamber",     default=None, choices=["house","senate","congress"])
    parser.add_argument("--year",        type=int, default=None)
    parser.add_argument("--force",       action="store_true", help="Re-convert even if output exists")
    parser.add_argument("--no-download", action="store_true",
                        help="Only convert manually placed files; never hit the network")
    args = parser.parse_args()

    state    = args.state.upper()
    chambers = [args.chamber] if args.chamber else ["house", "senate", "congress"]
    years    = [args.year]    if args.year    else sorted({y for (_s,_c,y) in TIGER_URLS if _s == state})

    print(f"\nBoundary fetch — {state}")
    print(f"Manual drop-folder: {MANUAL_DIR.resolve()}")
    print(f"Output:             data/states/{state}/{{chamber}}/{{year}}/boundaries.geojson\n")

    ok, failed = [], []
    for chamber in chambers:
        for year in years:
            print(f"── {chamber} {year} ──")
            success = fetch_boundary(state, chamber, year, args.force, args.no_download)
            (ok if success else failed).append(f"{chamber}/{year}")

    print(f"\n{'─'*50}")
    print(f"Done: {len(ok)} succeeded, {len(failed)} failed/skipped")
    if failed:
        print(f"Failed: {failed}")
        print(f"\nFor each failed item, place shapefiles in:")
        print(f"  {MANUAL_DIR.resolve()}/<STATE>_<chamber>_<year>/")
        print(f"Then re-run with --no-download to convert only.")
    if ok:
        print(f"\nNext: python scripts/fetch_elections.py --state {state}")


if __name__ == "__main__":
    main()
