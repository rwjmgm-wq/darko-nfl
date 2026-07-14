"""
V3: Per-season NFL start counts from nflverse play-by-play.

The survival model needs starts PER SEASON (person-period data), not career
totals. A "start" is proxied as a game with >= 10 pass attempts by the QB --
robust to mop-up relief appearances, unlike the old >= 1 attempt rule.

Output: data/processed/nfl_starts_by_season.csv
        (passer_id, passer_name, season, games_10att, games_any, attempts)

Matching to college QBs happens downstream by (F.Last name, draft year window),
with passer_player_id retained so ambiguous joins can be resolved.
"""

from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

ROOT = Path(__file__).parent.parent
SEASONS = range(2007, 2026)


def main():
    rows = []
    for season in SEASONS:
        print(f'[{season}] downloading...', flush=True)
        df = nfl.import_pbp_data([season], downcast=False, cache=False)
        p = df[df['pass'] == 1][['game_id', 'passer_player_id', 'passer_player_name']].dropna()
        per_game = p.groupby(['passer_player_id', 'passer_player_name', 'game_id']).size().reset_index(name='atts')
        agg = per_game.groupby(['passer_player_id', 'passer_player_name']).agg(
            games_10att=('atts', lambda s: (s >= 10).sum()),
            games_any=('atts', 'size'),
            attempts=('atts', 'sum'),
        ).reset_index()
        agg['season'] = season
        rows.append(agg)
        print(f'  {len(agg)} passers', flush=True)
        del df

    out = pd.concat(rows, ignore_index=True)
    out_path = ROOT / 'data' / 'processed' / 'nfl_starts_by_season.csv'
    out.to_csv(out_path, index=False)
    print(f'[OK] {len(out)} passer-seasons -> {out_path}')


if __name__ == '__main__':
    main()
