from sportsdataverse.cfb import load_cfb_pbp

df = load_cfb_pbp(seasons=[2016]).to_pandas()
teams = sorted(df['homeTeamName'].unique())

print('Looking for our QBs teams in 2016 data:')
search = ['North Carolina', 'Clemson', 'Texas Tech', 'Notre Dame', 'Iowa', 'Pittsburgh']
for s in search:
    matches = [t for t in teams if s.lower() in t.lower()]
    if matches:
        print(f'{s}: {matches}')
    else:
        print(f'{s}: NOT FOUND')

print('\nChecking drive.team.name column:')
if 'drive.team.name' in df.columns:
    drive_teams = sorted(df['drive.team.name'].dropna().unique()[:20])
    print('Sample drive team names:', drive_teams)
else:
    print('drive.team.name column does not exist!')
    print('Available columns with "team":', [c for c in df.columns if 'team' in c.lower()][:10])
