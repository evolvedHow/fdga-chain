"""
fetch_precinct_data.py — Download Georgia precinct data from MGGG-States.

MGGG (the GerryChain team) maintains pre-processed state files with:
  - Census VTD boundaries
  - 2020 Census population data
  - VEST election results already joined

This script downloads and prepares the Georgia file so it's ready
for build_graph.py.

Usage:
    uv run python scripts/fetch_precinct_data.py

Notes:
  - MGGG Georgia file: https://github.com/mggg-states/GA-shapefiles
  - Alternatively: use the Redistricting Data Hub (RDH) at
    https://redistrictingdatahub.org/state/georgia/
  - The RDH "Georgia 2020 VTDs with 2020 Census + VEST data" is the
    most complete ready-to-use source.

Column reference (MGGG/VEST convention):
  G20PREDBID  = Biden 2020 general
  G20PRERTRP  = Trump 2020 general
  G22GOVDBOS  = Abrams 2022 governor
  G22GOVRKMP  = Kemp 2022 governor
  TOTPOP20    = 2020 total population
  VAP20       = 2020 voting-age population
  BVAP20      = Black voting-age population
  HISP20      = Hispanic/Latino population
  HDIST       = enacted House district assignment
  SDIST       = enacted Senate district assignment
  CDIST       = enacted Congressional district assignment
"""

import sys
from pathlib import Path

DATA_RAW = Path("data/raw")


def main():
    print("Georgia precinct data setup guide")
    print("=" * 60)
    print()
    print("Option 1 (recommended): MGGG-States GitHub")
    print("  Repository: https://github.com/mggg-states/GA-shapefiles")
    print("  Download the ZIP, extract into data/raw/")
    print("  The shapefile will be named something like:")
    print("    GA_precincts20/GA_precincts20.shp")
    print()
    print("Option 2: Redistricting Data Hub")
    print("  URL: https://redistrictingdatahub.org/state/georgia/")
    print("  Download: 'Georgia Voting Tabulation Districts (VTDs) 2020'")
    print("  with election data joined.")
    print()
    print("Option 3: Use existing fdex data as a starting point")
    print("  Your fdex project has district-level GeoJSONs.")
    print("  These are district polygons, not precincts — GerryChain")
    print("  needs precinct-level data to recombine districts.")
    print()
    print("Once you have the shapefile, run:")
    print("  uv run python scripts/build_graph.py --chamber house")
    print()

    # Check if data already present
    existing = list(DATA_RAW.glob("*.shp"))
    if existing:
        print(f"Found existing shapefiles in data/raw/:")
        for f in existing:
            print(f"  {f.name}")
        print()
        print("Run build_graph.py to process them.")
    else:
        print("No shapefiles found in data/raw/ yet.")

    DATA_RAW.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
