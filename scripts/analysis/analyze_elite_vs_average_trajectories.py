"""
Analyze Elite vs Average Player Career Trajectories

Tests hypothesis: Do elite players plateau longer and decline slower?

Segments players by peak performance tier:
- Elite (peak > +1.0 z-score, top 15%)
- Above Avg (peak +0.5 to +1.0)
- Average (peak -0.5 to +0.5)
- Below Avg (peak < -0.5)

Analyzes:
1. How long each tier maintains peak performance
2. Decline rates after peak
3. Whether elite RBs decline as fast as average RBs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Load data
DATA_DIR = Path("data/production")

print("="*80)
print("ELITE vs AVERAGE CAREER TRAJECTORY ANALYSIS")
print("="*80)

# Load WR data
wr_files = sorted(DATA_DIR.glob("wr_composite_game_by_game_*.csv"))
wr_data = pd.read_csv(wr_files[-1])
print(f"\nLoaded {len(wr_data):,} WR game records")

# Load RB data
rb_files = sorted(DATA_DIR.glob("rb_composite_game_by_game_*.csv"))
rb_data = pd.read_csv(rb_files[-1])
print(f"Loaded {len(rb_data):,} RB game records")

def analyze_by_skill_tier(data, position_name, player_id_col):
    """Analyze career trajectories segmented by peak skill level."""

    print(f"\n{'='*80}")
    print(f"{position_name} - SKILL TIER ANALYSIS")
    print(f"{'='*80}")

    # Find each player's peak rating
    player_peaks = data.groupby(player_id_col).agg({
        'smoothed_composite': ['max', 'mean', 'count'],
        player_id_col.replace('_id', '_name'): 'first'
    })
    player_peaks.columns = ['peak_rating', 'avg_rating', 'total_games', 'player_name']
    player_peaks = player_peaks[player_peaks['total_games'] >= 30]  # Min 30 games

    # Define skill tiers based on peak rating
    def classify_tier(peak):
        if peak >= 1.0:
            return 'Elite (>+1.0)'
        elif peak >= 0.5:
            return 'Above Avg (+0.5 to +1.0)'
        elif peak >= -0.5:
            return 'Average (-0.5 to +0.5)'
        else:
            return 'Below Avg (<-0.5)'

    player_peaks['tier'] = player_peaks['peak_rating'].apply(classify_tier)

    # Print tier distribution
    print(f"\n{'='*60}")
    print(f"Player Distribution by Peak Performance Tier (30+ games)")
    print(f"{'='*60}")
    tier_counts = player_peaks['tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = (count / len(player_peaks)) * 100
        avg_peak = player_peaks[player_peaks['tier'] == tier]['peak_rating'].mean()
        print(f"{tier:30} {count:3d} players ({pct:5.1f}%) - Avg Peak: {avg_peak:+.3f}")

    # Merge tier info back to game data
    tier_map = player_peaks['tier'].to_dict()
    data['skill_tier'] = data[player_id_col].map(tier_map)
    data_with_tiers = data[data['skill_tier'].notna()].copy()

    # For each tier, analyze post-peak decline
    print(f"\n{'='*60}")
    print(f"POST-PEAK TRAJECTORY ANALYSIS")
    print(f"{'='*60}")

    tier_results = {}

    for tier in ['Elite (>+1.0)', 'Above Avg (+0.5 to +1.0)', 'Average (-0.5 to +0.5)', 'Below Avg (<-0.5)']:
        tier_players = player_peaks[player_peaks['tier'] == tier].index

        if len(tier_players) == 0:
            continue

        # For each player in tier, find their peak game and analyze decline
        post_peak_changes = []
        plateau_lengths = []

        for player_id in tier_players:
            player_games = data[data[player_id_col] == player_id].copy()
            player_games = player_games.sort_values('career_game_number')

            if len(player_games) < 30:
                continue

            # Find peak game (highest smoothed rating)
            peak_idx = player_games['smoothed_composite'].idxmax()
            peak_game_num = player_games.loc[peak_idx, 'career_game_number']
            peak_rating = player_games.loc[peak_idx, 'smoothed_composite']

            # Games after peak
            post_peak_games = player_games[player_games['career_game_number'] > peak_game_num]

            if len(post_peak_games) >= 10:  # Need 10+ games post-peak
                # Calculate decline rate (games per 0.1 z-score decline)
                first_post_peak = post_peak_games.iloc[0]['smoothed_composite']
                last_post_peak = post_peak_games.iloc[-1]['smoothed_composite']
                games_span = len(post_peak_games)

                rating_change = last_post_peak - first_post_peak
                post_peak_changes.append({
                    'player_name': player_games[player_id_col.replace('_id', '_name')].iloc[0],
                    'peak_rating': peak_rating,
                    'peak_game': peak_game_num,
                    'post_peak_games': games_span,
                    'rating_change': rating_change,
                    'decline_per_season': (rating_change / games_span) * 16 if games_span > 0 else 0
                })

                # Plateau length: games within 90% of peak
                plateau_threshold = peak_rating * 0.9
                plateau_games = player_games[player_games['smoothed_composite'] >= plateau_threshold]
                plateau_lengths.append(len(plateau_games))

        if len(post_peak_changes) > 0:
            changes_df = pd.DataFrame(post_peak_changes)
            avg_decline = changes_df['decline_per_season'].mean()
            median_decline = changes_df['decline_per_season'].median()
            avg_plateau = np.mean(plateau_lengths) if len(plateau_lengths) > 0 else 0

            # Count improvers vs decliners post-peak
            improvers = (changes_df['rating_change'] > 0).sum()
            decliners = (changes_df['rating_change'] < 0).sum()

            tier_results[tier] = {
                'n_players': len(post_peak_changes),
                'avg_decline_per_season': avg_decline,
                'median_decline_per_season': median_decline,
                'avg_plateau_games': avg_plateau,
                'pct_improve_post_peak': (improvers / len(changes_df)) * 100,
                'changes_df': changes_df
            }

            print(f"\n{tier}:")
            print(f"  Players analyzed: {len(post_peak_changes)}")
            print(f"  Avg plateau length: {avg_plateau:.1f} games ({avg_plateau/16:.1f} seasons)")
            print(f"  Post-peak decline: {avg_decline:+.3f} z-score/season (median: {median_decline:+.3f})")
            print(f"  % still improving post-peak: {(improvers / len(changes_df)) * 100:.1f}%")

            # Show example elite players
            if tier == 'Elite (>+1.0)' and len(changes_df) >= 5:
                print(f"\n  Elite {position_name} Post-Peak Patterns:")
                top_sustainers = changes_df.nlargest(5, 'decline_per_season')[
                    ['player_name', 'peak_rating', 'post_peak_games', 'decline_per_season']
                ]
                for _, row in top_sustainers.iterrows():
                    print(f"    {row['player_name']:25} Peak: {row['peak_rating']:+.3f}, "
                          f"{row['post_peak_games']} games post-peak, {row['decline_per_season']:+.3f}/season")

    # Statistical comparison: Elite vs Average
    if 'Elite (>+1.0)' in tier_results and 'Average (-0.5 to +0.5)' in tier_results:
        elite_decline = tier_results['Elite (>+1.0)']['avg_decline_per_season']
        avg_decline = tier_results['Average (-0.5 to +0.5)']['avg_decline_per_season']
        elite_plateau = tier_results['Elite (>+1.0)']['avg_plateau_games']
        avg_plateau = tier_results['Average (-0.5 to +0.5)']['avg_plateau_games']

        print(f"\n{'='*60}")
        print(f"ELITE vs AVERAGE COMPARISON")
        print(f"{'='*60}")
        print(f"Post-Peak Decline Rate:")
        print(f"  Elite: {elite_decline:+.3f} z-score/season")
        print(f"  Average: {avg_decline:+.3f} z-score/season")
        print(f"  Difference: {elite_decline - avg_decline:+.3f} ({((elite_decline - avg_decline) / abs(avg_decline) * 100) if avg_decline != 0 else 0:.1f}%)")

        print(f"\nPlateau Length:")
        print(f"  Elite: {elite_plateau:.1f} games ({elite_plateau/16:.1f} seasons)")
        print(f"  Average: {avg_plateau:.1f} games ({avg_plateau/16:.1f} seasons)")
        print(f"  Difference: +{elite_plateau - avg_plateau:.1f} games")

        # Test: Do elite players decline SLOWER (less negative) or FASTER (more negative)?
        if elite_decline > avg_decline:
            print(f"\n** FINDING: Elite {position_name}s decline SLOWER than average players **")
        else:
            print(f"\n** FINDING: Elite {position_name}s decline FASTER than average players (regression to mean) **")

    return tier_results

# Analyze both positions
print("\n\n")
wr_tier_results = analyze_by_skill_tier(wr_data, 'WR', 'receiver_player_id')

print("\n\n")
rb_tier_results = analyze_by_skill_tier(rb_data, 'RB', 'player_id')

# Cross-position comparison
print(f"\n{'='*80}")
print("POSITION COMPARISON: Elite Player Sustainability")
print(f"{'='*80}")

if 'Elite (>+1.0)' in wr_tier_results and 'Elite (>+1.0)' in rb_tier_results:
    wr_elite = wr_tier_results['Elite (>+1.0)']
    rb_elite = rb_tier_results['Elite (>+1.0)']

    print(f"\nElite WRs:")
    print(f"  Plateau: {wr_elite['avg_plateau_games']:.1f} games ({wr_elite['avg_plateau_games']/16:.1f} seasons)")
    print(f"  Post-peak decline: {wr_elite['avg_decline_per_season']:+.3f}/season")
    print(f"  % still improving: {wr_elite['pct_improve_post_peak']:.1f}%")

    print(f"\nElite RBs:")
    print(f"  Plateau: {rb_elite['avg_plateau_games']:.1f} games ({rb_elite['avg_plateau_games']/16:.1f} seasons)")
    print(f"  Post-peak decline: {rb_elite['avg_decline_per_season']:+.3f}/season")
    print(f"  % still improving: {rb_elite['pct_improve_post_peak']:.1f}%")

    plateau_diff = wr_elite['avg_plateau_games'] - rb_elite['avg_plateau_games']
    print(f"\n** Elite WRs sustain peak {plateau_diff:+.1f} games longer than elite RBs **")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")
print("Does elite talent = longer plateau?")
print("Does elite talent = slower decline?")
print("Or does position override skill level (RBs drop fast regardless)?")
print(f"{'='*80}")
