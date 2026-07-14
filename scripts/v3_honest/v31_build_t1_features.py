"""
A1 layers L5 + L6: season-pair features known at prediction time (offseason
before season Y+1).

L5a team_change   - QB's team in Y+1 differs from end of Y (free agency
                    precedes the season; known in spring).
L5b games_missed  - team games minus QB games in season Y (injury/bench proxy).
L6  sched_effect  - mean defense rating (as of END of season Y) of the Y+1
                    opponents actually on the QB's team's schedule. The rating
                    used for each defense is its def_rating at its FIRST game
                    of Y+1, which by construction uses only data through Y.
                    Positive = softer schedule = more observed EPA expected.

Output: data/processed/v31_t1_pairs.csv (all eras; train/test split downstream)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent


def main():
    df = pd.read_csv(ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'game_id']).reset_index(drop=True)

    # end-of-season-Y rating for each defense = its def_rating at its first game of Y+1
    first_def = (df.sort_values('game_date').groupby(['defteam', 'season'])['def_rating']
                 .first().rename('def_rating_start').reset_index())
    def_start = {(r.defteam, r.season): r.def_rating_start for r in first_def.itertuples()}

    # opponents faced by each team each season (the team's actual schedule)
    team_opps = (df.groupby(['posteam', 'season'])['defteam']
                 .apply(lambda s: list(s.unique())).to_dict())
    team_games = df.groupby(['posteam', 'season'])['game_id'].nunique().to_dict()

    rows = []
    for pid, g in df.groupby('passer_player_id'):
        g = g.sort_values('game_date')
        seasons = g.groupby('season').agg(plays=('plays', 'sum'), games=('game_id', 'nunique')).reset_index()
        for y in seasons['season']:
            gy = g[g.season == y]
            gy1 = g[g.season == y + 1]
            if gy['plays'].sum() < 150 or gy1['plays'].sum() < 150:
                continue

            team_y = gy['posteam'].iloc[-1]
            team_y1 = gy1['posteam'].iloc[0]

            # L6: schedule effect for the Y+1 team's actual schedule
            opps = team_opps.get((team_y1, y + 1), [])
            ratings = [def_start.get((o, y + 1), np.nan) for o in opps]
            ratings = [r for r in ratings if not np.isnan(r)]
            sched_effect = float(np.mean(ratings)) if ratings else 0.0

            # L5b: games missed in season Y
            tg = team_games.get((team_y, y), 16)
            games_missed = max(0, tg - gy['game_id'].nunique())

            rows.append({
                'passer_player_id': pid, 'y': y,
                'target': np.average(gy1['epa'], weights=gy1['plays']),
                'leaf_v3': gy['leaf_v3'].iloc[-1],
                'k_informed': gy['k_informed'].iloc[-1],
                'k_informed_var': gy['k_informed_var'].iloc[-1],
                'b1_expanding': gy['b1_expanding'].iloc[-1],
                'b4_last12': gy['b4_last12'].iloc[-1],
                'team_change': int(team_y != team_y1),
                'games_missed': games_missed,
                'sched_effect': sched_effect,
                'age_y1': gy['age'].iloc[-1] + 1 if pd.notna(gy['age'].iloc[-1]) else np.nan,
                'plays_y1': gy1['plays'].sum(),
            })

    out = pd.DataFrame(rows)
    out_dir = ROOT / 'data' / 'processed'
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / 'v31_t1_pairs.csv', index=False)
    print(f'[OK] {len(out)} season pairs -> v31_t1_pairs.csv')
    print(f'  team_change rate: {out.team_change.mean():.0%} | '
          f'games_missed mean: {out.games_missed.mean():.1f} | '
          f'sched_effect sd: {out.sched_effect.std():.4f}')


if __name__ == '__main__':
    main()
