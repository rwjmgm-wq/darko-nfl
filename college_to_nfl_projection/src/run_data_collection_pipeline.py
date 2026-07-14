"""
Master Data Collection Pipeline

Launches all data collection scripts with retry logic and tracks progress.
Run this to automatically collect all data needed for the comprehensive analysis.
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime
import pandas as pd


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def check_sp_plus_progress():
    """Check SP+ historical data collection progress"""
    root = get_project_root()
    sp_plus_dir = root / 'data' / 'processed' / 'sp_plus_historical'

    if not sp_plus_dir.exists():
        return 0, 21

    # Count collected years
    collected = 0
    for year in range(2003, 2024):
        year_file = sp_plus_dir / f'sp_plus_{year}.csv'
        if year_file.exists():
            try:
                df = pd.read_csv(year_file)
                if len(df) > 0:
                    collected += 1
            except:
                pass

    return collected, 21


def check_career_data_progress():
    """Check college career data collection progress"""
    root = get_project_root()
    career_dir = root / 'data' / 'processed' / 'full_college_careers'

    if not career_dir.exists():
        return 0, 124

    # Count collected QBs
    collected = set()
    for file in career_dir.glob('*_*.csv'):
        if file.name == 'career_summary.csv':
            continue

        try:
            df = pd.read_csv(file)
            if len(df) > 0:
                parts = file.stem.rsplit('_', 1)
                if len(parts) == 2:
                    collected.add(file.stem)
        except:
            pass

    return len(collected), 124


def check_nfl_outcomes_progress():
    """Check if NFL outcomes have been extracted"""
    root = get_project_root()
    outcomes_path = root / 'data' / 'processed' / 'nfl_outcomes_comprehensive.csv'

    if outcomes_path.exists():
        try:
            df = pd.read_csv(outcomes_path)
            return len(df), len(df)
        except:
            return 0, 124

    return 0, 124


def check_aggregations_progress():
    """Check if aggregations have been created"""
    root = get_project_root()
    agg_dir = root / 'data' / 'processed' / 'aggregated_stats'

    if not agg_dir.exists():
        return 0, 6

    methods = [
        'final_season',
        'career_average',
        'recency_weighted',
        'cumulative',
        'best_season',
        'volume_weighted'
    ]

    completed = 0
    for method in methods:
        method_file = agg_dir / f'stats_{method}.csv'
        if method_file.exists():
            try:
                df = pd.read_csv(method_file)
                if len(df) > 0:
                    completed += 1
            except:
                pass

    return completed, 6


def check_testing_progress():
    """Check if comprehensive testing has been done"""
    root = get_project_root()
    results_path = root / 'data' / 'processed' / 'comprehensive_model_results.csv'

    if results_path.exists():
        try:
            df = pd.read_csv(results_path)
            return len(df), len(df)
        except:
            return 0, 1

    return 0, 1


def print_status():
    """Print current pipeline status"""
    print("\n" + "="*80)
    print(f"PIPELINE STATUS - {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)

    # Phase 1: SP+ Historical Data
    sp_collected, sp_total = check_sp_plus_progress()
    sp_pct = (sp_collected / sp_total * 100) if sp_total > 0 else 0
    sp_status = "COMPLETE" if sp_collected == sp_total else "IN PROGRESS"

    print(f"\n  [1] SP+ Historical Data: {sp_collected}/{sp_total} years ({sp_pct:.0f}%) [{sp_status}]")

    # Phase 2: College Career Data
    career_collected, career_total = check_career_data_progress()
    career_pct = (career_collected / career_total * 100) if career_total > 0 else 0
    career_status = "COMPLETE" if career_collected == career_total else "IN PROGRESS"

    print(f"  [2] College Career Data: {career_collected}/{career_total} QBs ({career_pct:.0f}%) [{career_status}]")

    # Phase 3: NFL Outcomes
    nfl_collected, nfl_total = check_nfl_outcomes_progress()
    nfl_pct = (nfl_collected / nfl_total * 100) if nfl_total > 0 else 0
    nfl_status = "COMPLETE" if nfl_collected == nfl_total else "PENDING"

    print(f"  [3] NFL Outcomes: {nfl_collected}/{nfl_total} QBs ({nfl_pct:.0f}%) [{nfl_status}]")

    # Phase 4: Aggregations
    agg_collected, agg_total = check_aggregations_progress()
    agg_pct = (agg_collected / agg_total * 100) if agg_total > 0 else 0
    agg_status = "COMPLETE" if agg_collected == agg_total else "PENDING"

    print(f"  [4] Multi-Year Aggregations: {agg_collected}/{agg_total} methods ({agg_pct:.0f}%) [{agg_status}]")

    # Phase 5: Testing
    test_collected, test_total = check_testing_progress()
    test_status = "COMPLETE" if test_collected > 0 else "PENDING"

    print(f"  [5] Model Testing: [{test_status}]")

    # Overall progress
    phases_complete = sum([
        sp_collected == sp_total,
        career_collected == career_total,
        nfl_collected == nfl_total,
        agg_collected == agg_total,
        test_collected > 0
    ])

    print(f"\n  Overall Pipeline: {phases_complete}/5 phases complete")

    # Next actions
    print("\n  Next Actions:")
    if sp_collected < sp_total:
        print(f"    - SP+ fetch running in background (will retry every 10 min)")
    if career_collected < career_total:
        print(f"    - Need to collect {career_total - career_collected} more QB careers")
    elif nfl_collected < nfl_total:
        print(f"    - Ready to extract NFL outcomes")
    elif agg_collected < agg_total:
        print(f"    - Ready to create aggregation methods")
    elif test_collected == 0:
        print(f"    - Ready to run comprehensive testing")
    else:
        print(f"    - Pipeline complete! Run generate_comprehensive_report.py")


def main():
    """Main execution"""
    print("="*80)
    print("DATA COLLECTION PIPELINE ORCHESTRATOR")
    print("="*80)
    print("\nThis script will launch and monitor all data collection jobs")

    root = get_project_root()

    # Show initial status
    print_status()

    print("\n" + "="*80)
    print("LAUNCH OPTIONS")
    print("="*80)
    print("\n  You can launch the following scripts in separate terminals:")
    print("\n  Background jobs (with auto-retry):")
    print(f"    python src/fetch_historical_sp_plus_retry.py")
    print(f"    python src/collect_full_college_careers_retry.py")
    print(f"    python src/extract_nfl_outcomes_retry.py")

    print("\n  Sequential jobs (run after prerequisites complete):")
    print(f"    python src/create_multi_year_aggregations.py")
    print(f"    python src/test_comprehensive_models.py")
    print(f"    python src/generate_comprehensive_report.py")

    print("\n" + "="*80)
    print("RECOMMENDED EXECUTION")
    print("="*80)
    print("\n  Step 1: Launch background collectors in separate terminals")
    print("    Terminal 1: python src/fetch_historical_sp_plus_retry.py")
    print("    Terminal 2: python src/collect_full_college_careers_retry.py")
    print("    Terminal 3: python src/extract_nfl_outcomes_retry.py")

    print("\n  Step 2: Wait for data collection to complete")
    print("    (Monitor progress by re-running this script)")

    print("\n  Step 3: Once all data is collected, run sequential jobs:")
    print("    python src/create_multi_year_aggregations.py")
    print("    python src/test_comprehensive_models.py")
    print("    python src/generate_comprehensive_report.py")

    print("\n" + "="*80)
    print("MONITORING")
    print("="*80)
    print("\n  Re-run this script anytime to check progress:")
    print(f"    python src/run_data_collection_pipeline.py")

    print("\n  Background jobs will automatically retry every 30 minutes")
    print("  if they hit API errors, so you can leave them running overnight.")

    # Check if we can proceed to next steps
    sp_collected, sp_total = check_sp_plus_progress()
    career_collected, career_total = check_career_data_progress()
    nfl_collected, nfl_total = check_nfl_outcomes_progress()
    agg_collected, agg_total = check_aggregations_progress()

    print("\n" + "="*80)
    print("READY TO RUN")
    print("="*80)

    if career_collected == career_total and nfl_collected == nfl_total and agg_collected < agg_total:
        print("\n  College and NFL data complete!")
        print("  Ready to create aggregations:")
        print("    python src/create_multi_year_aggregations.py")

    elif agg_collected == agg_total and nfl_collected == nfl_total:
        print("\n  All data collection and aggregation complete!")
        print("  Ready to test models:")
        print("    python src/test_comprehensive_models.py")

    else:
        print("\n  Waiting for data collection to complete...")
        print(f"    Career data: {career_collected}/{career_total}")
        print(f"    NFL outcomes: {nfl_collected}/{nfl_total}")


if __name__ == "__main__":
    main()
