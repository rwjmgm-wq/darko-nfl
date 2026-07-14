import pandas as pd

df = pd.read_csv(r'C:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\college_to_nfl_projection\data\processed\merged_college_nfl.csv')

print('Opponents data availability:')
print(f'Total QBs: {len(df)}')
print(f'QBs with opponents data: {df["opponents"].notna().sum()}')

print('\nSample with opponents:')
sample = df[df['opponents'].notna()].head(3)
print(sample[['player_name', 'opponents']])

if len(sample) > 0:
    print(f'\nFirst QB opponents list:')
    opps = sample['opponents'].iloc[0].split('|')
    for i, opp in enumerate(opps[:10]):
        print(f'  {i+1}. {opp}')
