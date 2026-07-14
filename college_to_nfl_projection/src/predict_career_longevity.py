"""
Predict Career Longevity (85+ Starts)

Tests whether college stats can predict which QBs will have
long NFL careers (85+ starts = 5+ seasons as starter).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import nfl_data_py as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


def convert_to_nfl_name(full_name):
    """Convert full name to nfl_data_py abbreviated format"""
    parts = full_name.split()
    if len(parts) < 2:
        return full_name
    first_initial = parts[0][0]
    last_name = parts[-1]
    return f"{first_initial}.{last_name}"


def get_career_starts(player_name, draft_year):
    """
    Get total career starts for a QB

    Returns number of games started
    """
    nfl_name = convert_to_nfl_name(player_name)

    # Load from draft year through 2023
    seasons = list(range(draft_year, 2024))

    try:
        print(f"    Loading seasons {draft_year}-2023...", end=' ', flush=True)
        df_pbp = nfl.import_pbp_data(seasons, downcast=False)
    except Exception as e:
        print(f"[ERROR] {str(e)[:50]}")
        return 0

    # Filter to this QB's plays
    qb_plays = df_pbp[
        (
            (df_pbp['passer_player_name'] == nfl_name) |
            (df_pbp['passer_player_name'] == player_name)
        ) &
        (df_pbp['pass'] == 1)
    ].copy()

    if len(qb_plays) == 0:
        print("[--] No plays found")
        return 0

    # Count unique games
    games = qb_plays[['season', 'week', 'game_id']].drop_duplicates()
    total_games = len(games)

    print(f"[OK] {total_games} games")
    return total_games


def main():
    """Main execution"""
    print("="*80)
    print("CAREER LONGEVITY PREDICTION (85+ STARTS)")
    print("="*80)
    print("\nTesting if college stats predict QB career longevity")

    root = get_project_root()

    # Load QB list
    print("\n[1] Loading QB list...")
    qb_path = root / 'data' / 'processed' / 'merged_with_sp_plus.csv'
    df_qbs = pd.read_csv(qb_path)
    print(f"    [OK] {len(df_qbs)} QBs")

    # Extract career starts for each QB
    print("\n[2] Extracting career starts for each QB...")
    career_data = []

    for _, qb in df_qbs.iterrows():
        player_name = qb['player_name']
        draft_year = qb['draft_year']

        print(f"  {player_name:25s} (drafted {draft_year}): ", end='', flush=True)

        career_starts = get_career_starts(player_name, draft_year)

        career_data.append({
            'player_name': player_name,
            'draft_year': draft_year,
            'draft_pick': qb['draft_pick'],
            'career_starts': career_starts,
            'reached_85_starts': career_starts >= 85
        })

    df_careers = pd.DataFrame(career_data)
    print(f"\n[OK] Extracted career data for {len(df_careers)} QBs")

    # Save career data
    print("\n[3] Saving career longevity data...")
    output_path = root / 'data' / 'processed' / 'qb_career_starts.csv'
    df_careers.to_csv(output_path, index=False)
    print(f"    [OK] Saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("CAREER LONGEVITY DISTRIBUTION")
    print("="*80)

    total_qbs = len(df_careers)
    reached_85 = df_careers['reached_85_starts'].sum()

    print(f"\nTotal QBs: {total_qbs}")
    print(f"Reached 85+ starts: {reached_85} ({reached_85/total_qbs*100:.1f}%)")
    print(f"Did not reach 85 starts: {total_qbs - reached_85} ({(total_qbs-reached_85)/total_qbs*100:.1f}%)")

    print(f"\nCareer starts statistics:")
    print(f"  Mean: {df_careers['career_starts'].mean():.1f}")
    print(f"  Median: {df_careers['career_starts'].median():.1f}")
    print(f"  Max: {df_careers['career_starts'].max():.0f}")
    print(f"  Min: {df_careers['career_starts'].min():.0f}")

    # Load college stats
    print("\n[4] Loading college comprehensive stats...")
    college_path = root / 'data' / 'processed' / 'qb_comprehensive_stats.csv'
    df_college = pd.read_csv(college_path)

    # Merge
    print("\n[5] Merging college stats with career data...")
    df_merged = df_college.merge(
        df_careers[['player_name', 'career_starts', 'reached_85_starts']],
        on='player_name',
        how='inner'
    )
    print(f"    [OK] {len(df_merged)} QBs with complete data")

    # 7-feature model
    features = [
        'attempts',
        'big_play_rate',
        'explosive_rate',
        'high_leverage_epa',
        'red_zone_epa',
        'third_down_epa',
        'opp_adj_epa'
    ]

    # Prepare data
    X = df_merged[features]
    y = df_merged['reached_85_starts'].astype(int)

    valid_mask = X.notna().all(axis=1) & y.notna()
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    df_valid = df_merged[valid_mask].copy()

    print(f"\n    Analysis sample: {len(df_valid)} QBs")
    print(f"    Positives (85+ starts): {y_valid.sum()}")
    print(f"    Negatives (<85 starts): {len(y_valid) - y_valid.sum()}")

    # Train logistic regression model
    print("\n[6] Training logistic regression model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y_valid)

    # Predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # Evaluation metrics
    accuracy = accuracy_score(y_valid, y_pred)

    # Handle case where all predictions are same class
    try:
        precision = precision_score(y_valid, y_pred)
        recall = recall_score(y_valid, y_pred)
    except:
        precision = 0
        recall = 0

    try:
        auc = roc_auc_score(y_valid, y_pred_proba)
    except:
        auc = 0.5

    print(f"\n    Model Performance:")
    print(f"      Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"      Precision: {precision:.3f}")
    print(f"      Recall:    {recall:.3f}")
    print(f"      AUC:       {auc:.3f}")

    # Feature importances
    print(f"\n    Feature Coefficients:")
    for feature, coef in zip(features, model.coef_[0]):
        print(f"      {feature:25s}: {coef:+.3f}")
    print(f"      Intercept: {model.intercept_[0]:.3f}")

    # Add predictions to dataframe
    df_valid['longevity_probability'] = y_pred_proba
    df_valid['predicted_85_starts'] = y_pred

    # Rank by probability
    df_valid = df_valid.sort_values('longevity_probability', ascending=False)

    # Show results
    print("\n" + "="*80)
    print("TOP 10 PREDICTED FOR LONG CAREERS (by model)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Player':<25} {'Probability':>12} {'Actual':>10} {'Starts':>8} {'Draft':>8}")
    print("-"*80)

    for i, (_, row) in enumerate(df_valid.head(10).iterrows(), 1):
        player = row['player_name']
        prob = row['longevity_probability']
        actual = "YES" if row['reached_85_starts'] else "NO"
        starts = int(row['career_starts'])
        draft = int(row['draft_pick'])

        print(f"{i:<6} {player:<25} {prob:>11.1%} {actual:>10} {starts:>8} #{draft:<7}")

    print("\n" + "="*80)
    print("BOTTOM 10 PREDICTED FOR LONG CAREERS (by model)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Player':<25} {'Probability':>12} {'Actual':>10} {'Starts':>8} {'Draft':>8}")
    print("-"*80)

    bottom_10 = df_valid.tail(10)
    for i, (_, row) in enumerate(bottom_10.iterrows(), 1):
        rank = len(df_valid) - 10 + i
        player = row['player_name']
        prob = row['longevity_probability']
        actual = "YES" if row['reached_85_starts'] else "NO"
        starts = int(row['career_starts'])
        draft = int(row['draft_pick'])

        print(f"{rank:<6} {player:<25} {prob:>11.1%} {actual:>10} {starts:>8} #{draft:<7}")

    # Comparison to draft capital
    print("\n" + "="*80)
    print("DRAFT CAPITAL VS MODEL COMPARISON")
    print("="*80)

    # Draft pick baseline
    from scipy.stats import pointbiserialr
    corr_draft, p_draft = pointbiserialr(df_valid['reached_85_starts'], -df_valid['draft_pick'])

    print(f"\nDraft Pick Predictive Power:")
    print(f"  Point-biserial correlation: r = {corr_draft:+.3f}")
    print(f"  P-value: {p_draft:.4f}")

    # Model correlation
    corr_model, p_model = pointbiserialr(df_valid['reached_85_starts'], df_valid['longevity_probability'])

    print(f"\nCollege Model Predictive Power:")
    print(f"  Point-biserial correlation: r = {corr_model:+.3f}")
    print(f"  P-value: {p_model:.4f}")

    if abs(corr_model) > abs(corr_draft):
        improvement = (abs(corr_model) - abs(corr_draft)) / abs(corr_draft) * 100
        print(f"\n  [SUCCESS] Model {improvement:+.1f}% better than draft capital!")
    else:
        decline = (abs(corr_draft) - abs(corr_model)) / abs(corr_draft) * 100
        print(f"\n  [--] Model {decline:.1f}% worse than draft capital")

    # Confusion matrix analysis
    print("\n" + "="*80)
    print("PREDICTION ACCURACY BREAKDOWN")
    print("="*80)

    true_positives = ((y_valid == 1) & (y_pred == 1)).sum()
    false_positives = ((y_valid == 0) & (y_pred == 1)).sum()
    true_negatives = ((y_valid == 0) & (y_pred == 0)).sum()
    false_negatives = ((y_valid == 1) & (y_pred == 0)).sum()

    print(f"\nConfusion Matrix:")
    print(f"  True Positives (correctly predicted long career):  {true_positives}")
    print(f"  False Positives (predicted long, actually short):  {false_positives}")
    print(f"  True Negatives (correctly predicted short career): {true_negatives}")
    print(f"  False Negatives (predicted short, actually long):  {false_negatives}")

    # Save results
    print("\n[7] Saving longevity predictions...")
    pred_path = root / 'data' / 'processed' / 'longevity_predictions.csv'
    df_valid.to_csv(pred_path, index=False)
    print(f"    [OK] Saved to: {pred_path}")

    print("\n" + "="*80)
    print("LONGEVITY PREDICTION COMPLETE")
    print("="*80)
    print(f"\nModel AUC: {auc:.3f}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"College model correlation: r = {corr_model:+.3f}")


if __name__ == "__main__":
    main()
