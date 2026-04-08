"""
fetch_elections.py — Download and process historical election results by district.

Sources:
  - MIT Election Lab (congressional): https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
  - Georgia SOS (state legislative): manual download from sos.ga.gov
  - Redistricting Data Hub (pre-joined to districts): redistrictingdatahub.org

Output:
  data/states/{STATE}/{chamber}/{year}/elections.json
  Format:
  {
    "state": "GA",
    "chamber": "house",
    "cycle_year": 2011,
    "election_year": 2012,   // first general election on these maps
    "total_votes": 1234567,
    "total_dem_votes": 600000,
    "total_rep_votes": 600000,
    "districts": [
      {
        "district_id": "1",
        "dem_votes": 45000,
        "rep_votes": 60000,
        "total_votes": 108000,
        "dem_pct": 0.417,
        "winner": "R",
        "margin": 0.139
      },
      ...
    ]
  }

Usage:
  uv run python scripts/fetch_elections.py --state GA
  uv run python scripts/fetch_elections.py --state GA --chamber congress --year 2011

NOTE: MIT Election Lab congressional data requires a one-time manual download
from the Harvard Dataverse (free, requires account):
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
Save the CSV as: data/raw/mit_house_elections.csv
                 data/raw/mit_senate_elections.csv

For Georgia state legislative data, download from:
  https://sos.ga.gov/page/elections-division-georgia-secretary-state
Or use pre-joined files from:
  https://redistrictingdatahub.org/state/georgia/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")

# Mapping of redistricting cycle year → which election year to use as the
# representative election (first general election on the new maps)
CYCLE_TO_ELECTION_YEAR = {
    2001: 2002,
    2005: 2006,
    2011: 2012,
    2021: 2022,
}


# ---------------------------------------------------------------------------
# MIT Election Lab parser (congressional)
# ---------------------------------------------------------------------------

def load_mit_congressional(state: str, election_year: int) -> pd.DataFrame | None:
    """
    Load MIT Election Lab 1976–2022 House of Representatives election data.
    CSV columns include: year, state_po, district, candidate, party, candidatevotes, totalvotes, stage
    """
    csv_path = RAW_DIR / "mit_house_elections.csv"
    if not csv_path.exists():
        print(f"  MISSING: {csv_path}")
        print("  Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2")
        print("  File: 1976-2022-house.csv → rename to mit_house_elections.csv")
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.lower()

    mask = (
        (df["year"] == election_year) &
        (df["state_po"] == state.upper()) &
        (df["stage"] == "GEN")
    )
    filtered = df[mask].copy()
    if filtered.empty:
        print(f"  No MIT election data for {state} congressional {election_year}")
        return None

    # Aggregate by district + party
    dem_mask = filtered["party"].str.upper().isin(["DEMOCRAT", "DEMOCRATIC"])
    rep_mask = filtered["party"].str.upper().isin(["REPUBLICAN"])

    records = []
    for district_id in filtered["district"].unique():
        d = filtered[filtered["district"] == district_id]
        dem_votes = int(d[dem_mask & (d["district"] == district_id)]["candidatevotes"].sum())
        rep_votes = int(d[rep_mask & (d["district"] == district_id)]["candidatevotes"].sum())
        total = int(d["totalvotes"].iloc[0]) if "totalvotes" in d.columns else dem_votes + rep_votes
        two_party = dem_votes + rep_votes
        dem_pct = dem_votes / two_party if two_party > 0 else 0.5
        records.append({
            "district_id": str(int(district_id)),
            "dem_votes": dem_votes,
            "rep_votes": rep_votes,
            "total_votes": total,
            "dem_pct": round(dem_pct, 4),
            "winner": "D" if dem_votes > rep_votes else "R",
            "margin": round(abs(dem_pct - 0.5) * 2, 4),
        })

    return pd.DataFrame(records).sort_values("district_id")


# ---------------------------------------------------------------------------
# Georgia SOS / RDH state legislative parser
# ---------------------------------------------------------------------------

def load_ga_state_leg(chamber: str, election_year: int) -> pd.DataFrame | None:
    """
    Load pre-joined Georgia state legislative election results.
    Expects a CSV at data/raw/ga_{chamber}_elections_{year}.csv

    Column format (flexible — we try several common column names):
      district, dem_votes, rep_votes   (minimal)
      OR: district_id, dem, rep        (alternative)
    """
    for fname in [
        f"ga_{chamber}_elections_{election_year}.csv",
        f"ga_{chamber}_{election_year}.csv",
        f"{chamber}_{election_year}.csv",
    ]:
        csv_path = RAW_DIR / fname
        if csv_path.exists():
            break
    else:
        print(f"  MISSING: data/raw/ga_{chamber}_elections_{election_year}.csv")
        print("  Download from: https://redistrictingdatahub.org/state/georgia/")
        print("  Or: https://sos.ga.gov/page/elections-division-georgia-secretary-state")
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    # Normalize column names
    for col in ["district", "dist", "district_id", "district_number"]:
        if col in df.columns:
            df = df.rename(columns={col: "district_id"})
            break

    for col in ["dem_votes", "dem", "democratic", "democratic_votes"]:
        if col in df.columns:
            df = df.rename(columns={col: "dem_votes"})
            break

    for col in ["rep_votes", "rep", "republican", "republican_votes"]:
        if col in df.columns:
            df = df.rename(columns={col: "rep_votes"})
            break

    required = ["district_id", "dem_votes", "rep_votes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing columns {missing} in {csv_path}")
        print(f"  Found: {list(df.columns)}")
        return None

    df["district_id"] = df["district_id"].astype(str).str.lstrip("0").str.strip()
    df["dem_votes"] = pd.to_numeric(df["dem_votes"], errors="coerce").fillna(0).astype(int)
    df["rep_votes"] = pd.to_numeric(df["rep_votes"], errors="coerce").fillna(0).astype(int)
    df["total_votes"] = df["dem_votes"] + df["rep_votes"]

    df["two_party"] = df["dem_votes"] + df["rep_votes"]
    df["dem_pct"] = (df["dem_votes"] / df["two_party"]).where(df["two_party"] > 0, 0.5).round(4)
    df["winner"] = df.apply(lambda r: "D" if r["dem_votes"] > r["rep_votes"] else "R", axis=1)
    df["margin"] = ((df["dem_pct"] - 0.5).abs() * 2).round(4)

    return df[["district_id", "dem_votes", "rep_votes", "total_votes", "dem_pct", "winner", "margin"]]


# ---------------------------------------------------------------------------
# Assemble and write elections.json
# ---------------------------------------------------------------------------

def build_elections_json(state: str, chamber: str, cycle_year: int) -> Path | None:
    election_year = CYCLE_TO_ELECTION_YEAR.get(cycle_year)
    if not election_year:
        print(f"  No election year mapping for cycle {cycle_year}")
        return None

    print(f"  Loading {state} {chamber} {election_year} election results…")

    if chamber == "congress":
        df = load_mit_congressional(state, election_year)
    else:
        df = load_ga_state_leg(chamber, election_year)

    if df is None or df.empty:
        return None

    districts = df.to_dict(orient="records")
    total_dem = int(df["dem_votes"].sum())
    total_rep = int(df["rep_votes"].sum())
    total = int(df["total_votes"].sum())

    data = {
        "state":          state.upper(),
        "chamber":        chamber.lower(),
        "cycle_year":     cycle_year,
        "election_year":  election_year,
        "total_votes":    total,
        "total_dem_votes": total_dem,
        "total_rep_votes": total_rep,
        "dem_seat_share": round(len(df[df["winner"] == "D"]) / len(df), 4),
        "rep_seat_share": round(len(df[df["winner"] == "R"]) / len(df), 4),
        "dem_vote_share": round(total_dem / (total_dem + total_rep), 4) if (total_dem + total_rep) > 0 else 0.5,
        "num_districts":  len(df),
        "districts":      districts,
    }

    out_dir = Path(f"data/states/{state.upper()}/{chamber.lower()}/{cycle_year}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "elections.json"
    out_path.write_text(json.dumps(data, indent=2))
    size_kb = out_path.stat().st_size / 1024
    print(f"  ✓ {out_path}  ({len(districts)} districts, {size_kb:.1f} kB)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and process historical election results by district."
    )
    parser.add_argument("--state",   required=True, help="State abbreviation (e.g. GA)")
    parser.add_argument("--chamber", default=None,
                        choices=["house", "senate", "congress"],
                        help="Specific chamber (default: all)")
    parser.add_argument("--year",    type=int, default=None,
                        help="Specific redistricting cycle year (default: all)")
    args = parser.parse_args()

    state = args.state.upper()
    chambers = [args.chamber] if args.chamber else ["house", "senate", "congress"]
    all_cycle_years = sorted(CYCLE_TO_ELECTION_YEAR.keys())
    years = [args.year] if args.year else all_cycle_years

    print(f"\nFetching election results for {state}")
    print(f"Chambers: {chambers}")
    print(f"Cycle years: {years}\n")

    print("Data source requirements:")
    print("  Congressional: data/raw/mit_house_elections.csv")
    print("    → https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2")
    print("  State leg:     data/raw/ga_{chamber}_elections_{year}.csv")
    print("    → https://redistrictingdatahub.org/state/ga/")
    print()

    errors = []
    for chamber in chambers:
        for year in years:
            print(f"── {chamber} {year} ──")
            try:
                result = build_elections_json(state, chamber, year)
                if result is None:
                    errors.append(f"{chamber}/{year}")
            except Exception as e:
                print(f"  ERROR: {e}")
                errors.append(f"{chamber}/{year}")
            print()

    if errors:
        print(f"Missing or failed: {errors}")
        print("Download the source files listed above, then re-run.")
    else:
        print("Done. Run scripts/compute_metrics.py --state " + state + " next.")


if __name__ == "__main__":
    main()
