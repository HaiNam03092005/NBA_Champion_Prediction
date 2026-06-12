import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import specialized modules
from models import train_stacking_baseline, train_voting_baseline
from simulator import simulate_conference_bracket, simulate_monte_carlo_bo7

warnings.filterwarnings("ignore")

os.makedirs('results', exist_ok=True)

# 1.DOWNLOAD AND PREPARE DATA
df = pd.read_csv('data/processed/model_ready_dataset.csv')
df.fillna(0, inplace=True)

try:
    h2h_lookup = pd.read_csv('data/raw/nba_api_h2h_matrix.csv')
except FileNotFoundError:
    h2h_lookup = pd.DataFrame(
        columns=['Season_Year', 'TEAM_NAME', 'OPPONENT_NAME', 'H2H_Win_Rate']
    )

# Extract ALL potential features to incorporate into the self-filtering machine
absolute_features = [
    'Has_Top5_MVP', 'All_NBA_Count', 'Prev_Year_Playoff_Round',
    'Age_Diff', 'Elite_Defense_Flag', 'Two_Way_Score', 'eFG_Diff',
    'Superstar_Impact_Score', 'Playoff_Core_VORP_Share', 'Core_Fatigue_MP',
    'Bench_Tactical_BPM', 'Is_Dynasty_Profile', 'Elo_Score',
    'Diff_PTS', 'Diff_eFG', 'Diff_REB', 'Diff_TOV'
]
all_possible_features = [
    col for col in df.columns 
    if col.endswith('_Rank') or col in absolute_features
]
all_possible_features = [f for f in all_possible_features if f in df.columns]

# Separate the Train/Test series by season
train_data = df[
    (df['Season_Year'] >= 2000) & (df['Season_Year'] <= 2020)
].copy()
test_data = df[
    (df['Season_Year'] >= 2021) & (df['Season_Year'] <= 2025)
].copy()

# Filter the teams into the Playoffs to use them as a springboard to find Feature
train_pool = train_data[train_data['Is_Playoff'] == 1].copy()

X_raw_train = train_pool[all_possible_features]
# The goal is to select features based on the ability to predict the winning team (Target_Champ)
y_raw_train = train_pool['Target_Champ'] 


print("\nAI is automatically detecting the optimal set of features")
base_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

selector = RFECV(
    estimator=base_estimator,
    step=1,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1
)
selector.fit(X_raw_train, y_raw_train)

# Extract a list of columns retained by the AI
features = list(np.array(all_possible_features)[selector.support_])

print(f"AI filtered {len(all_possible_features)} down to "
      f"{len(features)} The most outstanding features")

# Data normalization is based on a list of features selected by AI
scaler = StandardScaler()
scaler.fit(train_data[features])
X_train_pool_scaled = scaler.transform(train_pool[features])

# 2.TRAIN 2 MODELS (Using new optimized features)
print("TRAINING MODEL 1: VOTING ENSEMBLE")
vote_top4, vote_top2, vote_champ = train_voting_baseline(
    X_train_pool_scaled, train_pool['Target_Top4'],
    train_pool['Target_Top2'], train_pool['Target_Champ']
)

print("TRAINING MODEL 2: STANDARD STACKING")
stack_top4, stack_top2, stack_champ = train_stacking_baseline(
    X_train_pool_scaled, train_pool['Target_Top4'],
    train_pool['Target_Top2'], train_pool['Target_Champ']
)

# 3.MODEL EVALUATION
def evaluate_model(m_top4, m_top2, m_champ):
    total_top4, total_top2, total_champ = 0, 0, 0
    max_top4, max_top2, max_champ = 0, 0, 0
    for year in sorted(test_data['Season_Year'].unique()):
        year_data = test_data[test_data['Season_Year'] == year].copy()
        
        actual_top4 = year_data[
            year_data['Target_Top4'] == 1
        ]['TEAM_NAME'].tolist()
        actual_top2 = year_data[
            year_data['Target_Top2'] == 1
        ]['TEAM_NAME'].tolist()
        actual_champ_row = year_data[year_data['Target_Champ'] == 1]
        
        if not actual_champ_row.empty:
            actual_champ = actual_champ_row.iloc[0]['TEAM_NAME']
        else:
            actual_champ = "UNKNOWN"
        
        playoffs = year_data[year_data['Is_Playoff'] == 1].copy()
        east = playoffs[playoffs['Conference'] == 'East'].sort_values(
            by=['W', 'NRtg'], ascending=[False, False]
        )
        west = playoffs[playoffs['Conference'] == 'West'].sort_values(
            by=['W', 'NRtg'], ascending=[False, False]
        )
        
        if len(east) != 8 or len(west) != 8:
            continue
            
        east['Seed'], west['Seed'] = range(1, 9), range(1, 9)
        p16 = pd.concat([east, west])
        X_test = scaler.transform(p16[features])
        
        p16['Prob_Top4'] = m_top4.predict_proba(X_test)[:, 1]
        p16['Prob_Top2'] = m_top2.predict_proba(X_test)[:, 1]
        p16['Prob_Champ'] = m_champ.predict_proba(X_test)[:, 1]
        
        east_top4, east_champ = simulate_conference_bracket(
            p16[p16['Conference'] == 'East'], 'Prob_Top4', 'Prob_Top2',
            h2h_lookup
        )
        west_top4, west_champ = simulate_conference_bracket(
            p16[p16['Conference'] == 'West'], 'Prob_Top4', 'Prob_Top2',
            h2h_lookup
        )
        
        predicted_top4 = east_top4 + west_top4
        predicted_top2 = [east_champ['TEAM_NAME'], west_champ['TEAM_NAME']]
        predicted_champ = simulate_monte_carlo_bo7(
            east_champ, west_champ, 'Prob_Champ', h2h_lookup
        )['TEAM_NAME']
            
        if len(actual_top4) > 0:
            total_top4 += len(set(predicted_top4) & set(actual_top4))
            max_top4 += 4
        if len(actual_top2) > 0:
            total_top2 += len(set(predicted_top2) & set(actual_top2))
            max_top2 += 2
        if actual_champ != "UNKNOWN":
            total_champ += 1 if predicted_champ == actual_champ else 0
            max_champ += 1

    acc_top4 = (total_top4 / max_top4) * 100 if max_top4 > 0 else 0
    acc_top2 = (total_top2 / max_top2) * 100 if max_top2 > 0 else 0
    acc_champ = (total_champ / max_champ) * 100 if max_champ > 0 else 0
    return acc_top4, acc_top2, acc_champ


v_top4, v_top2, v_champ = evaluate_model(vote_top4, vote_top2, vote_champ)

s_top4, s_top2, s_champ = evaluate_model(stack_top4, stack_top2, stack_champ)

print("COMPARISON TABLE OF 2 MODELS: VOTING vs. STACKING")
print(f"| {'Evaluation Criteria':<20} | {'Model 1 (Voting)':<20} | "
      f"{'Model 2 (Stacking)':<20} |")
print("-" * 70)
print(f"| {'Predicting the Top 4':<20} | {v_top4:>19.1f}% | {s_top4:>19.1f}% |")
print(f"| {'NBA Finals Predictions':<20} | {v_top2:>19.1f}% | {s_top2:>19.1f}% |")
print(f"| {'Champion prediction':<20} | {v_champ:>19.1f}% | {s_champ:>19.1f}% |")

# 4.DRAW THE CHART AND SAVE IT TO THE 'RESULTS' FOLDER
labels = ['Top 4', 'Top 2 (Finals)', 'Nhà Vô Địch']
voting_scores = [v_top4, v_top2, v_champ]
stacking_scores = [s_top4, s_top2, s_champ]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width/2, voting_scores, width, label='Voting', color='#4c72b0')
plt.bar(x + width/2, stacking_scores, width, label='Stacking', color='#dd8452')

plt.ylabel('Accuracy(%)', fontweight='bold')
plt.title('PERFORMANCE COMPARISON: VOTING VS. STACKING', fontweight='bold')
plt.xticks(x, labels)
plt.ylim(0, 100)
plt.legend()

for i, v in enumerate(voting_scores):
    plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
for i, v in enumerate(stacking_scores):
    plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/model_performance_comparison.png', dpi=300)
plt.close()

#  Plotting a Feature Importance chart based on the selected set
rf_estimator = vote_champ.named_estimators_['rf']
importances = rf_estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('THE IMPORTANCE LEVEL OF FEATURES IS CHOSEN BY AI', fontweight='bold')
plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('results/top15_feature_importance.png', dpi=300)
plt.close()

print("1.results/model_performance_comparison.png")
print("2.results/top15_feature_importance.png")