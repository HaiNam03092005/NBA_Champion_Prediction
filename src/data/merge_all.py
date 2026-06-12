import os
import pandas as pd

# 1.Defining data file paths
PATH_MASTER_INPUT    = 'data/interim/teams_dataset.csv'             # Original Team File
PATH_PLAYER_INPUT    = 'data/interim/player_data_playoff_ready.csv'  # Healthy/Injured Player File
PATH_API_DIFF_INPUT  = 'data/raw/nba_api_diff_features.csv'          # NBA API Matchup Index File
PATH_FINAL_OUTPUT    = 'data/processed/final_complete_dataset.csv'   # Final complete file

# STEP 1: READ AND ANALYZE PLAYER FITNESS AND LINEUP DEPTH
if not os.path.exists(PATH_PLAYER_INPUT):
    print(f"Error: Player data file not found '{PATH_PLAYER_INPUT}'")
    exit()

df_players = pd.read_csv(PATH_PLAYER_INPUT)
team_season_features = []

for (year, team), group in df_players.groupby(['Season_Year', 'TEAM_NAME']):
    # Sort players by VORP contribution in descending order
    group_sorted = group.sort_values(by='VORP', ascending=False).reset_index(drop=True)
    
    total_team_vorp = group_sorted['VORP'].sum()
    playoff_ready_players = len(group)
    team_avg_obpm = group['OBPM'].mean()
    team_avg_dbpm = group['DBPM'].mean()
    
    # Get the statistics for the Top 7 players who carry their teams (For Playoff Round)
    top7 = group_sorted.head(7)
    top7_vorp = top7['VORP'].sum()
    top7_vorp_share = top7_vorp / total_team_vorp if total_team_vorp != 0 else 0
    
    # Calculate the level of time overload in the Top 5 core players
    top5_minutes = group_sorted.sort_values(by='Regular_Season_MP', ascending=False).head(5)
    total_core_mp = top5_minutes['Regular_Season_MP'].sum() 
    
    # Measuring the quality of the reserve squad (Rotation positions 8, 9, 10)
    bench = group_sorted.iloc[7:10]
    bench_tactical_bpm = (bench['OBPM'] + bench['DBPM']).mean() if len(bench) > 0 else 0
    
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
cols_to_round = ['Team_Avg_OBPM', 'Team_Avg_DBPM', 'Team_Total_VORP', 'Playoff_Core_VORP_Share', 'Bench_Tactical_BPM']
df_players_agg[cols_to_round] = df_players_agg[cols_to_round].round(3)

# STEP 2: ĐỌC VÀ CHUẨN HÓA ĐỒNG BỘ FILE ĐỘI BÓNG GỐC
print("\nStep 2: Loading and normalizing file structure teams_dataset.csv")
if not os.path.exists(PATH_MASTER_INPUT):
    print(f"ERROR: Raw team master file not found '{PATH_MASTER_INPUT}'")
    exit()

df_master_orig = pd.read_csv(PATH_MASTER_INPUT)

# Standardize column header names
if 'Team' in df_master_orig.columns: df_master_orig.rename(columns={'Team': 'TEAM_NAME'}, inplace=True)
if 'Year' in df_master_orig.columns: df_master_orig.rename(columns={'Year': 'Season_Year'}, inplace=True)
if 'Season' in df_master_orig.columns: df_master_orig.rename(columns={'Season': 'Season_Year'}, inplace=True)

# Synchronized dictionary of NBA club abbreviations/history avoids key discrepancies during mergers
team_name_mapping = {
    'Los Angeles Clippers': 'LA Clippers',
    'Charlotte Bobcats': 'Charlotte Hornets',
    'New Jersey Nets': 'Brooklyn Nets',
    'New Orleans/Oklahoma City Hornets': 'New Orleans Hornets'
}
df_master_orig['TEAM_NAME'] = df_master_orig['TEAM_NAME'].replace(team_name_mapping)
df_players_agg['TEAM_NAME'] = df_players_agg['TEAM_NAME'].replace(team_name_mapping)

# STEP 3: PERFORM TEAM AND PLAYER INFORMATION MIXING
print("\nStep 3: Team Stats and Player Features data are being assembled")
df_final_complete = pd.merge(df_master_orig, df_players_agg, on=['TEAM_NAME', 'Season_Year'], how='left')

# Enter 0 if that team is missing detailed player analysis data
cols_to_fillna = ['Playoff_Ready_Players', 'Team_Total_VORP', 'Team_Avg_OBPM', 'Team_Avg_DBPM', 
                  'Playoff_Core_VORP_Share', 'Core_Fatigue_MP', 'Bench_Tactical_BPM']
df_final_complete[cols_to_fillna] = df_final_complete[cols_to_fillna].fillna(0)

# STEP 4: ADD COUNTER-ATTRIBUTE STATISTICS FROM THE NBA API RAW FILE
print("\nStep 4: Integrating the matchup metrics from the NBA API")
if os.path.exists(PATH_API_DIFF_INPUT):
    df_api_diff = pd.read_csv(PATH_API_DIFF_INPUT)
    df_api_diff['TEAM_NAME'] = df_api_diff['TEAM_NAME'].replace(team_name_mapping)
    
    # Prevent column duplication if the summary file is run repeat
    target_api_cols = ['Diff_PTS', 'Diff_eFG', 'Diff_REB', 'Diff_TOV']
    df_final_complete = df_final_complete.drop(columns=[c for c in target_api_cols if c in df_final_complete.columns], errors='ignore')
    
    # Proceed with merging
    df_final_complete = pd.merge(df_final_complete, df_api_diff, on=['Season_Year', 'TEAM_NAME'], how='left')
    df_final_complete[target_api_cols] = df_final_complete[target_api_cols].fillna(0)
else:
    print("Warning: The file 'nba_api_diff_features.csv' was not found. API index integration was skipped")


# STEP 5: EXPORT THE COMPLETE DATASET FILE FOR TRAINING
print(f"\nStep 5: Packaging and saving the aggregated dataset '{PATH_FINAL_OUTPUT}'")
os.makedirs(os.path.dirname(PATH_FINAL_OUTPUT), exist_ok=True)
df_final_complete.to_csv(PATH_FINAL_OUTPUT, index=False)

print(f"The complete dataset has the following size: {df_final_complete.shape[0]} row, {df_final_complete.shape[1]} column")