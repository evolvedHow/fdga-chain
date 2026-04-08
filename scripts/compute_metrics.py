"""
compute_metrics.py — Compute redistricting fairness metrics from elections.json.

Metrics produced:
  - efficiency_gap       (signed: + = favors R, - = favors D)
  - wasted_dem_votes     (packed + cracked)
  - wasted_rep_votes
  - dem_seat_share       (fraction of seats won by D)
  - dem_vote_share       (fraction of total votes for D)
  - seats_votes_curve    (at various hypothetical vote share levels)
  - mean_median          (avg Dem vote share − median Dem vote share)
  - partisan_bias        (seats won at exactly 50% statewide vote)
  - responsiveness       (d(seat_share)/d(vote_share) near 50%)

Output:
  data/states/{STATE}/{chamber}/{year}/metrics.json

Usage:
  uv run python scripts/compute_metrics.py --state GA
  uv run python scripts/compute_metrics.py --state GA --chamber congress --year 2011
"""

import argparse
import json
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Core metric calculations
# ---------------------------------------------------------------------------

def compute_wasted_votes(dem_votes: int, rep_votes: int) -> tuple[int, int]:
    """
    Wasted votes per district.
    Winning party: votes above majority threshold are wasted (excess margin).
    Losing party: all votes are wasted.
    """
    total = dem_votes + rep_votes
    threshold = total // 2 + 1  # minimum to win

    if dem_votes > rep_votes:
        wasted_dem = dem_votes - threshold
        wasted_rep = rep_votes
    elif rep_votes > dem_votes:
        wasted_rep = rep_votes - threshold
        wasted_dem = dem_votes
    else:
        # Tie — split waste equally (rare)
        wasted_dem = dem_votes // 2
        wasted_rep = rep_votes // 2

    return max(0, wasted_dem), max(0, wasted_rep)


def efficiency_gap(total_wasted_dem: int, total_wasted_rep: int, total_votes: int) -> float:
    """
    Efficiency gap = (wasted_D - wasted_R) / total_votes.
    Positive = more D waste = Republican advantage.
    """
    if total_votes == 0:
        return 0.0
    return (total_wasted_dem - total_wasted_rep) / total_votes


def mean_median(dem_vote_shares: list[float]) -> float:
    """
    Mean - median of district-level Dem vote shares.
    Positive = Democrats over-perform in mean (packed), under in median (cracked).
    """
    if not dem_vote_shares:
        return 0.0
    n = len(dem_vote_shares)
    mean_val = sum(dem_vote_shares) / n
    sorted_shares = sorted(dem_vote_shares)
    if n % 2 == 1:
        median_val = sorted_shares[n // 2]
    else:
        median_val = (sorted_shares[n // 2 - 1] + sorted_shares[n // 2]) / 2
    return mean_val - median_val


def seats_votes_curve(
    dem_vote_shares: list[float],
    swing_range: tuple[float, float] = (-0.15, 0.15),
    num_points: int = 31,
) -> list[dict]:
    """
    Simulate what seat share Democrats would get at various hypothetical
    statewide vote shares by applying a uniform swing to each district.

    Returns a list of {vote_share, seat_share} points.
    """
    step = (swing_range[1] - swing_range[0]) / (num_points - 1)
    results = []
    for i in range(num_points):
        swing = swing_range[0] + i * step
        adjusted = [min(1.0, max(0.0, s + swing)) for s in dem_vote_shares]
        dem_seats = sum(1 for s in adjusted if s > 0.5)
        actual_vote_share = sum(adjusted) / len(adjusted) if adjusted else 0.5
        results.append({
            "vote_share": round(actual_vote_share, 4),
            "seat_share": round(dem_seats / len(dem_vote_shares), 4),
            "dem_seats": dem_seats,
        })
    return results


def partisan_bias(seats_votes: list[dict]) -> float:
    """
    At 50% statewide vote share, how many seats does the Dem party win?
    Bias = that seat share - 0.5. Positive = D-favored.
    """
    # Find the point closest to 50% vote share
    if not seats_votes:
        return 0.0
    closest = min(seats_votes, key=lambda p: abs(p["vote_share"] - 0.5))
    return round(closest["seat_share"] - 0.5, 4)


def responsiveness(seats_votes: list[dict], window: float = 0.05) -> float:
    """
    Rate of change of seat share with respect to vote share near 50%.
    Higher = more responsive (competitive) system.
    """
    near_50 = [p for p in seats_votes if abs(p["vote_share"] - 0.5) <= window]
    if len(near_50) < 2:
        return 0.0
    near_50.sort(key=lambda p: p["vote_share"])
    # Simple linear regression slope
    xs = [p["vote_share"] for p in near_50]
    ys = [p["seat_share"] for p in near_50]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    return round(num / den, 4) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_metrics_for_cycle(
    state: str, chamber: str, cycle_year: int
) -> Path | None:
    elections_path = (
        Path(f"data/states/{state.upper()}/{chamber.lower()}/{cycle_year}")
        / "elections.json"
    )
    if not elections_path.exists():
        print(f"  MISSING: {elections_path}")
        print("  Run: uv run python scripts/fetch_elections.py --state " + state)
        return None

    elections = json.loads(elections_path.read_text())
    districts = elections.get("districts", [])
    if not districts:
        print(f"  ERROR: No district data in {elections_path}")
        return None

    # --- Per-district wasted votes ---
    total_wasted_dem = 0
    total_wasted_rep = 0
    total_votes = 0
    dem_vote_shares = []
    district_details = []

    for d in districts:
        dv = d.get("dem_votes", 0)
        rv = d.get("rep_votes", 0)
        tv = dv + rv
        wd, wr = compute_wasted_votes(dv, rv)
        total_wasted_dem += wd
        total_wasted_rep += wr
        total_votes += tv
        if tv > 0:
            dem_vote_shares.append(dv / tv)
        district_details.append({
            **d,
            "wasted_dem": wd,
            "wasted_rep": wr,
        })

    # --- Aggregate metrics ---
    eg = efficiency_gap(total_wasted_dem, total_wasted_rep, total_votes)
    mm = mean_median(dem_vote_shares)
    sv_curve = seats_votes_curve(dem_vote_shares)
    pb = partisan_bias(sv_curve)
    resp = responsiveness(sv_curve)

    dem_seats = sum(1 for d in districts if d.get("winner") == "D")
    rep_seats = sum(1 for d in districts if d.get("winner") == "R")
    num_districts = len(districts)

    # Interpretation tier (for display)
    def eg_tier(v):
        av = abs(v)
        if av < 0.05: return "competitive"
        if av < 0.08: return "mild"
        return "strong"

    metrics = {
        "state":            state.upper(),
        "chamber":          chamber.lower(),
        "cycle_year":       cycle_year,
        "election_year":    elections.get("election_year"),
        "num_districts":    num_districts,
        "total_votes":      total_votes,

        # Core fairness metrics
        "efficiency_gap":    round(eg, 5),
        "eg_direction":      "R" if eg > 0 else "D",
        "eg_tier":           eg_tier(eg),
        "eg_description":    (
            f"Republicans wasted {abs(eg)*100:.1f}% fewer votes than Democrats" if eg > 0
            else f"Democrats wasted {abs(eg)*100:.1f}% fewer votes than Republicans"
        ),

        "mean_median":       round(mm, 5),
        "partisan_bias":     pb,
        "responsiveness":    resp,

        # Seat / vote shares
        "dem_seats":         dem_seats,
        "rep_seats":         rep_seats,
        "dem_seat_share":    round(dem_seats / num_districts, 4) if num_districts else 0,
        "rep_seat_share":    round(rep_seats / num_districts, 4) if num_districts else 0,
        "dem_vote_share":    elections.get("dem_vote_share", 0),
        "rep_vote_share":    round(1 - elections.get("dem_vote_share", 0), 4),

        # Wasted votes
        "total_wasted_dem":  total_wasted_dem,
        "total_wasted_rep":  total_wasted_rep,
        "wasted_dem_pct":    round(total_wasted_dem / total_votes, 5) if total_votes else 0,
        "wasted_rep_pct":    round(total_wasted_rep / total_votes, 5) if total_votes else 0,

        # Seats-votes curve (31 points from -15% to +15% swing)
        "seats_votes_curve": sv_curve,

        # Per-district details (for map tooltips)
        "district_details":  district_details,
    }

    out_dir = Path(f"data/states/{state.upper()}/{chamber.lower()}/{cycle_year}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, separators=(",", ":")))
    size_kb = out_path.stat().st_size / 1024
    print(f"  ✓ {out_path}  ({size_kb:.1f} kB)")
    print(f"    EG: {eg:+.3f} ({eg_tier(eg)}, favors {'R' if eg>0 else 'D'})")
    print(f"    Seats: D {dem_seats} / R {rep_seats}  |  Votes: D {elections.get('dem_vote_share',0)*100:.1f}%")
    print(f"    Mean-median: {mm:+.3f}  |  Partisan bias: {pb:+.3f}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CYCLE_YEARS = [2001, 2005, 2011, 2021]


def main():
    parser = argparse.ArgumentParser(
        description="Compute redistricting fairness metrics from elections.json files."
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
    years = [args.year] if args.year else CYCLE_YEARS

    print(f"\nComputing metrics for {state}")
    print(f"Chambers: {chambers}")
    print(f"Cycle years: {years}\n")

    errors = []
    for chamber in chambers:
        for year in years:
            print(f"── {chamber} {year} ──")
            try:
                result = compute_metrics_for_cycle(state, chamber, year)
                if result is None:
                    errors.append(f"{chamber}/{year}")
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                errors.append(f"{chamber}/{year}")
            print()

    if errors:
        print(f"Failed or skipped: {errors}")
        print("Make sure elections.json files exist (run fetch_elections.py first).")
    else:
        print("Done. Metrics written to data/states/{state}/{chamber}/{year}/metrics.json")
        print("\nAPI endpoint: GET /api/states/" + state + "/{chamber}/metrics/{year}")


if __name__ == "__main__":
    main()
