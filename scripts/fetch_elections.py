"""
fetch_elections.py — Download and process Georgia election results by district.

Source: OpenElections GA — https://github.com/openelections/openelections-data-ga
All data is freely downloadable without registration.

Output:
  data/states/{STATE}/{chamber}/{year}/elections.json

Usage:
  uv run python scripts/fetch_elections.py --state GA
  uv run python scripts/fetch_elections.py --state GA --chamber congress --year 2011
"""

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# OpenElections GA URLs and office labels
# ---------------------------------------------------------------------------

# Maps (cycle_year) → the first general election run on those maps
CYCLE_TO_ELECTION_YEAR = {
    2001: 2002,
    2005: 2006,
    2011: 2012,
    2021: 2022,
}

# Maps (state, election_year) → raw CSV URL from openelections-data-ga
OE_URLS = {
    ("GA", 2002): "https://raw.githubusercontent.com/openelections/openelections-data-ga/master/2002/20021105__ga__general.csv",
    ("GA", 2006): "https://raw.githubusercontent.com/openelections/openelections-data-ga/master/2006/20061107__ga__general.csv",
    ("GA", 2012): "https://raw.githubusercontent.com/openelections/openelections-data-ga/master/2012/20121106__ga__general.csv",
    ("GA", 2022): "https://raw.githubusercontent.com/openelections/openelections-data-ga/master/2022/20221108__ga__general__precinct.csv",
}

# Maps chamber → the office label(s) used in OpenElections CSVs
CHAMBER_OFFICES = {
    "house":    ["State House"],
    "senate":   ["State Senate"],
    "congress": ["U.S. House"],
}

# Columns that count as "votes" (sum across all columns present)
VOTE_COLS = ["votes", "election_day_votes", "advanced_votes",
             "absentee_by_mail_votes", "provisional_votes"]


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def fetch_csv(url: str) -> pd.DataFrame:
    print(f"  ↓ {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fdga-chain/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read().decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(data), low_memory=False)


# Cache downloaded CSVs per (state, election_year)
_csv_cache: dict[tuple, pd.DataFrame] = {}


def get_oe_csv(state: str, election_year: int) -> pd.DataFrame | None:
    key = (state.upper(), election_year)
    if key in _csv_cache:
        return _csv_cache[key]

    url = OE_URLS.get(key)
    if not url:
        print(f"  No OpenElections URL configured for {state} {election_year}")
        return None

    try:
        df = fetch_csv(url)
        df.columns = df.columns.str.lower().str.strip()
        _csv_cache[key] = df
        return df
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return None


# ---------------------------------------------------------------------------
# Parse election results for a chamber
# ---------------------------------------------------------------------------

def parse_results(df: pd.DataFrame, chamber: str, state: str) -> pd.DataFrame | None:
    offices = CHAMBER_OFFICES.get(chamber, [])

    # Filter to the right office
    office_mask = df["office"].isin(offices)
    subset = df[office_mask].copy()

    if subset.empty:
        print(f"  No rows found for offices {offices}")
        return None

    # Normalize party
    subset["party_norm"] = subset["party"].str.upper().str.strip()

    # Sum all available vote columns
    vote_col_present = [c for c in VOTE_COLS if c in subset.columns]
    if not vote_col_present:
        print(f"  ERROR: no vote columns found. Available: {list(subset.columns)}")
        return None

    subset["votes_cast"] = subset[vote_col_present].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).sum(axis=1)

    # Normalize district
    if "district" not in subset.columns:
        print("  ERROR: no 'district' column")
        return None

    subset["district_id"] = (
        subset["district"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.lstrip("0")
        .str.strip()
    )
    subset = subset[subset["district_id"].notna() & (subset["district_id"] != "")]

    records = []
    for dist_id, grp in subset.groupby("district_id"):
        dem = grp[grp["party_norm"].isin(["DEMOCRAT", "DEMOCRATIC"])]["votes_cast"].sum()
        rep = grp[grp["party_norm"] == "REPUBLICAN"]["votes_cast"].sum()
        total = grp["votes_cast"].sum()
        two_party = dem + rep
        dem_pct = dem / two_party if two_party > 0 else 0.5
        records.append({
            "district_id": str(dist_id),
            "dem_votes":   int(dem),
            "rep_votes":   int(rep),
            "total_votes": int(total),
            "dem_pct":     round(float(dem_pct), 4),
            "winner":      "D" if dem > rep else "R",
            "margin":      round(abs(float(dem_pct) - 0.5) * 2, 4),
        })

    if not records:
        print("  No districts parsed")
        return None

    result = pd.DataFrame(records)
    # Natural-sort district IDs numerically
    result["_sort"] = pd.to_numeric(result["district_id"], errors="coerce")
    result = result.sort_values("_sort").drop(columns=["_sort"])
    return result


# ---------------------------------------------------------------------------
# Build and write elections.json
# ---------------------------------------------------------------------------

def build_elections_json(state: str, chamber: str, cycle_year: int) -> bool:
    election_year = CYCLE_TO_ELECTION_YEAR.get(cycle_year)
    if not election_year:
        print(f"  No election year mapped for cycle {cycle_year}")
        return False

    out_path = Path(f"data/states/{state.upper()}/{chamber.lower()}/{cycle_year}/elections.json")
    if out_path.exists():
        print(f"  ✓ Already exists (use --force to re-fetch)")
        return True

    df_raw = get_oe_csv(state, election_year)
    if df_raw is None:
        return False

    df = parse_results(df_raw, chamber, state)
    if df is None or df.empty:
        return False

    districts = df.to_dict(orient="records")
    total_dem = int(df["dem_votes"].sum())
    total_rep = int(df["rep_votes"].sum())
    total = int(df["total_votes"].sum())
    two_party_total = total_dem + total_rep
    n_districts = len(df)
    dem_seats = int((df["winner"] == "D").sum())

    data = {
        "state":           state.upper(),
        "chamber":         chamber.lower(),
        "cycle_year":      cycle_year,
        "election_year":   election_year,
        "num_districts":   n_districts,
        "total_votes":     total,
        "total_dem_votes": total_dem,
        "total_rep_votes": total_rep,
        "dem_vote_share":  round(total_dem / two_party_total, 4) if two_party_total else 0.5,
        "dem_seat_share":  round(dem_seats / n_districts, 4) if n_districts else 0.5,
        "rep_seat_share":  round((n_districts - dem_seats) / n_districts, 4) if n_districts else 0.5,
        "districts":       districts,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    size_kb = out_path.stat().st_size / 1024
    print(f"  ✓ {out_path}  ({n_districts} districts, {size_kb:.1f} kB)")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download GA election results from OpenElections and build elections.json"
    )
    parser.add_argument("--state",   required=True, help="State abbreviation (e.g. GA)")
    parser.add_argument("--chamber", default=None,
                        choices=["house", "senate", "congress"],
                        help="Specific chamber (default: all)")
    parser.add_argument("--year",    type=int, default=None,
                        help="Specific redistricting cycle year (default: all)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-download even if elections.json already exists")
    args = parser.parse_args()

    state = args.state.upper()
    chambers = [args.chamber] if args.chamber else ["house", "senate", "congress"]
    all_years = sorted(CYCLE_TO_ELECTION_YEAR.keys())
    years = [args.year] if args.year else all_years

    if args.force:
        for chamber in chambers:
            for year in years:
                p = Path(f"data/states/{state}/{chamber}/{year}/elections.json")
                if p.exists():
                    p.unlink()

    print(f"\nElection data — {state}  (source: OpenElections)")
    print(f"Chambers: {chambers}   Cycles: {years}\n")

    errors = []
    for chamber in chambers:
        for year in years:
            print(f"── {chamber} {year} ──")
            try:
                ok = build_elections_json(state, chamber, year)
                if not ok:
                    errors.append(f"{chamber}/{year}")
            except Exception as e:
                print(f"  ERROR: {e}")
                errors.append(f"{chamber}/{year}")
            print()

    if errors:
        print(f"Failed: {errors}")
        sys.exit(1)
    else:
        print(f"Done. Run: uv run python scripts/compute_metrics.py --state {state}")


if __name__ == "__main__":
    main()
