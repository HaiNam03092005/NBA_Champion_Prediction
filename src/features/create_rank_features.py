import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1.DEFINITION OF FILE PATH
# This file directly reads the merged result from merge_all.py
INPUT_FILE = 'data/processed/final_complete_dataset.csv'
OUTPUT_FILE = 'data/processed/model_ready_dataset.csv'


# 2.READ ORIGINAL DATA
print(f"Step 1: Reading data from '{INPUT_FILE}'")
if not os.path.exists(INPUT_FILE):
    print(f"Error: File not found '{INPUT_FILE}'. Please run merge_all.py before")
    exit()

df = pd.read_csv(INPUT_FILE)

# 3. [ADDITIONAL INFORMATION FROM LUKE-LITE] HANDLING ELO RATING
if 'Elo_Score' not in df.columns and all(c in df.columns for c in ['W', 'L', 'SRS']):
    win_pct = df['W'] / (df['W'] + df['L']).replace(0, 0.5)
    df['Elo_Score'] = 1500 + (win_pct - 0.5) * 300 + df['SRS'] * 15
elif 'Elo_Score' in df.columns:
    print("The 'Elo_Score' stat is already available in the data")

# 4. CALCULATING THE GOLDEN AGE (AGE DIFF)
if 'Age' in df.columns and 'Target_Champ' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    champ_mean_age = df[df['Target_Champ'] == 1]['Age'].mean()
    if pd.isna(champ_mean_age): champ_mean_age = 28.0 
    
    df['Age_Diff'] = abs(df['Age'] - champ_mean_age)
    df['Age_Diff_Rank'] = df.groupby('Season_Year')['Age_Diff'].rank(ascending=True, method='min')

# 5.CALCULATING BASE STANDINGS & COUNTER-RANKINGS
print("Step 3: Calculating Core Rank, Stamina & H2H Stats")
rank_metrics = {
    # Basic Team Stats
    'W': False, 'SRS': False, 'NRtg': False, 'ORtg': False, 
    'DRtg': True, 'MOV': False, 'TS%': False,
    'Team_Total_VORP': False, 'Team_Avg_OBPM': False, 'Team_Avg_DBPM': False,
    
    # Fitness Index & Team Formation
    'Playoff_Core_VORP_Share': False,  
    'Bench_Tactical_BPM': False,       
    'Core_Fatigue_MP': True,           
    
    # Stat of improved performance
    'Elo_Score': False,
    
    # RANKING OF H2HFEATURES (Included in merge_all)
    'Diff_PTS': False,   # The higher the score difference -> The better the rank (Rank 1)
    'Diff_eFG': False,   # The higher the basket performance difference -> The better the ranking
    'Diff_REB': False,   # Winning with more possession -> Better ranking
    'Diff_TOV': True     # Fewer turnovers than the opponent (Negative difference) -> Better ranking (True)
}

for col, is_ascending in rank_metrics.items():
    if col in df.columns:
        df[f"{col}_Rank"] = df.groupby('Season_Year')[col].rank(ascending=is_ascending, method='min')

# 6.REAL-LIFE BASKETBALL RULES (DOMAIN KNOWLEDGE)
print("Step 4: Initialize Domain Knowledge variables (Elite Defense, Two-Way, eFG_Diff, Superstar)")

if 'DRtg_Rank' in df.columns:
    df['Elite_Defense_Flag'] = (df['DRtg_Rank'] <= 11).astype(int)

if 'ORtg_Rank' in df.columns and 'DRtg_Rank' in df.columns:
    df['Two_Way_Score'] = df['ORtg_Rank'] + df['DRtg_Rank']
    df['Two_Way_Rank'] = df.groupby('Season_Year')['Two_Way_Score'].rank(ascending=True, method='min')

efg_off = 'Offense Four Factors_eFG%'
efg_def = 'Defense Four Factors_eFG%'
if efg_off in df.columns and efg_def in df.columns:
    df['eFG_Diff'] = df[efg_off] - df[efg_def]
    df['eFG_Diff_Rank'] = df.groupby('Season_Year')['eFG_Diff'].rank(ascending=False, method='min')

if all(c in df.columns for c in ['Has_Top5_MVP', 'All_NBA_Count', 'Team_Total_VORP']):
    df['Superstar_Impact_Score'] = df['Team_Total_VORP'] + (df['Has_Top5_MVP'] * 3) + df['All_NBA_Count']
    df['Superstar_Impact_Rank'] = df.groupby('Season_Year')['Superstar_Impact_Score'].rank(ascending=False, method='min')

if 'W' in df.columns and 'Conference' in df.columns:
    df['Conf_Seed'] = df.groupby(['Season_Year', 'Conference'])['W'].rank(ascending=False, method='min')

# 7.DBSCAN DYNASTY CLUSTERS
features_for_clustering = ['SRS', 'NRtg', 'TS%', 'Superstar_Impact_Score']
existing_clust_features = [c for c in features_for_clustering if c in df.columns]

if len(existing_clust_features) == len(features_for_clustering):
    X_cluster = df[existing_clust_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X_cluster)

    dbscan = DBSCAN(eps=0.6, min_samples=3)
    df['Dynasty_Cluster'] = dbscan.fit_predict(X_scaled)

    df['Is_Dynasty_Profile'] = (df['Dynasty_Cluster'] != -1).astype(int)
    print(f"Number of teams included in the Dynasty profile: {df['Is_Dynasty_Profile'].sum()} team")
else:
    print("WARNING: DBSCAN cannot be run due to missing one of the core feature fields")


# 8.PREPARE DATA FOR MODELING
print(f"\nStep 5: Packaging the final data into a file '{OUTPUT_FILE}'")
if 'Conf_Seed' in df.columns:
    df = df.sort_values(by=['Season_Year', 'Conference', 'Conf_Seed'], ascending=[False, True, True])

df.to_csv(OUTPUT_FILE, index=False)