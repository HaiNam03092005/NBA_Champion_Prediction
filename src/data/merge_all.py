import os
import pandas as pd

# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN CÁC FILE
PATH_MASTER_INPUT  = 'data/interim/teams_dataset.csv'             # Teams file
PATH_PLAYER_INPUT  = 'data/interim/player_data_playoff_ready.csv'  # File players ready for play-off
PATH_FINAL_OUTPUT  = 'data/processed/final_complete_dataset.csv' 

print("Final data merging process (TEAM + PLAYERS + FATIGUE)")

# 2. Read and calculate players and fitness data
print("Step 1: Loading and analyzing Playoff teams data (Rotation & Fatigue)")
if not os.path.exists(PATH_PLAYER_INPUT):
    print(f"Error: Player data file not found '{PATH_PLAYER_INPUT}'")
    exit()

df_players = pd.read_csv(PATH_PLAYER_INPUT)
team_season_features = []

# Analyzing deep lineup of the teams
for (year, team), group in df_players.groupby(['Season_Year', 'TEAM_NAME']):
    # Sort players by their Voting Value Proposition (VORP) from highest to lowest
    group_sorted = group.sort_values(by='VORP', ascending=False).reset_index(drop=True)
    
    # Old basic stats
    total_team_vorp = group_sorted['VORP'].sum()
    playoff_ready_players = len(group)
    team_avg_obpm = group['OBPM'].mean()
    team_avg_dbpm = group['DBPM'].mean()
    
    # New advaced stats
    # 1. Get data on the Top 7 key players (Playoff Rotation)
    top7 = group_sorted.head(7)
    top7_vorp = top7['VORP'].sum()
    top7_vorp_share = top7_vorp / total_team_vorp if total_team_vorp != 0 else 0
    
    # 2. Calculate the stamina/overload (Fatigue) of the 5 most played-mintue players
    top5_minutes = group_sorted.sort_values(by='Regular_Season_MP', ascending=False).head(5)
    total_core_mp = top5_minutes['Regular_Season_MP'].sum() 
    
    # 3. Quality of the reserve squad (Players in positions 8, 9, and 10)
    bench = group_sorted.iloc[7:10]
    bench_tactical_bpm = (bench['OBPM'] + bench['DBPM']).mean() if len(bench) > 0 else 0
    
    # Package all the stats
    team_season_features.append({
        'Season_Year': year,
        'TEAM_NAME': team,
        'Playoff_Ready_Players': playoff_ready_players,
        'Team_Total_VORP': total_team_vorp,
        'Team_Avg_OBPM': team_avg_obpm,
        'Team_Avg_DBPM': team_avg_dbpm,
        'Playoff_Core_VORP_Share': top7_vorp_share,
        'Core_Fatigue_MP': total_core_mp,
        'Bench_Tactical_BPM': bench_tactical_bpm
    })

df_players_agg = pd.DataFrame(team_season_features)

# Round the averages/percentages for a nicer look
cols_to_round = ['Team_Avg_OBPM', 'Team_Avg_DBPM', 'Team_Total_VORP', 'Playoff_Core_VORP_Share', 'Bench_Tactical_BPM']
df_players_agg[cols_to_round] = df_players_agg[cols_to_round].round(3)

print(f"The depth and stamina have been calculated {df_players_agg.shape[0]} basketball teams")

# 3. Read AND normalize the teams dataset file
print("Step 2: Loading dataset teams_dataset.csv")
if not os.path.exists(PATH_MASTER_INPUT):
    print(f"Error: Original file not found '{PATH_MASTER_INPUT}'")
    exit()

df_master_orig = pd.read_csv(PATH_MASTER_INPUT)

# Normalize column names
if 'Team' in df_master_orig.columns:
    df_master_orig.rename(columns={'Team': 'TEAM_NAME'}, inplace=True)
if 'Year' in df_master_orig.columns:
    df_master_orig.rename(columns={'Year': 'Season_Year'}, inplace=True)
if 'Season' in df_master_orig.columns:
    df_master_orig.rename(columns={'Season': 'Season_Year'}, inplace=True)

# Standard map dictionary of team names
team_name_mapping = {
    'Los Angeles Clippers': 'LA Clippers',
    'Charlotte Bobcats': 'Charlotte Hornets',
    'New Jersey Nets': 'Brooklyn Nets',
    'New Orleans/Oklahoma City Hornets': 'New Orleans Hornets'
}
df_master_orig['TEAM_NAME'] = df_master_orig['TEAM_NAME'].replace(team_name_mapping)
df_players_agg['TEAM_NAME'] = df_players_agg['TEAM_NAME'].replace(team_name_mapping)

# 4. Merge data & export files
print("Step 3: Proceed with assembly Master Dataset and Player Features")
df_final_complete = pd.merge(df_master_orig, df_players_agg, on=['TEAM_NAME', 'Season_Year'], how='left')

# Fill in 0 for teams with missing player data
cols_to_fillna = [
    'Playoff_Ready_Players', 'Team_Total_VORP', 'Team_Avg_OBPM', 'Team_Avg_DBPM', 
    'Playoff_Core_VORP_Share', 'Core_Fatigue_MP', 'Bench_Tactical_BPM'
]
df_final_complete[cols_to_fillna] = df_final_complete[cols_to_fillna].fillna(0)

print(f"Step 4: Writing all data to a file '{PATH_FINAL_OUTPUT}'")
os.makedirs(os.path.dirname(PATH_FINAL_OUTPUT), exist_ok=True)
df_final_complete.to_csv(PATH_FINAL_OUTPUT, index=False)

print(f"Complete merging")
print(f"Add more three stats: Core_Share, Fatigue_MP, Bench_BPM")
print(f"Size of final Dataset: {df_final_complete.shape[0]} rows, {df_final_complete.shape[1]} columns.")
print(f"The final file is saved: '{PATH_FINAL_OUTPUT}'")