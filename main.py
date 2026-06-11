import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import các mô-đun chuyên biệt
from models import train_stacking_baseline, train_voting_baseline
from simulator import simulate_conference_bracket, simulate_monte_carlo_bo7

warnings.filterwarnings("ignore")

os.makedirs('results', exist_ok=True)

# ==========================================================
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================================
print("📂 Đang tải dữ liệu từ model_ready_dataset.csv...")
df = pd.read_csv('data/processed/model_ready_dataset.csv')
df.fillna(0, inplace=True)

try:
    h2h_lookup = pd.read_csv('data/raw/nba_api_h2h_matrix.csv')
except FileNotFoundError:
    h2h_lookup = pd.DataFrame(
        columns=['Season_Year', 'TEAM_NAME', 'OPPONENT_NAME', 'H2H_Win_Rate']
    )

# Lấy TOÀN BỘ các đặc trưng tiềm năng để đưa vào máy tự lọc
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

# Tách tập Train/Test theo năm
train_data = df[
    (df['Season_Year'] >= 2000) & (df['Season_Year'] <= 2020)
].copy()
test_data = df[
    (df['Season_Year'] >= 2021) & (df['Season_Year'] <= 2025)
].copy()

# Lọc các đội vào Playoff để làm bàn đạp tìm Feature
train_pool = train_data[train_data['Is_Playoff'] == 1].copy()

X_raw_train = train_pool[all_possible_features]
# Mục tiêu chọn đặc trưng dựa trên khả năng đoán đội Vô Địch (Target_Champ)
y_raw_train = train_pool['Target_Champ'] 

# ----------------------------------------------------------
# 🤖 MACHINE LEARNING TỰ ĐỘNG DÒ TÌM FEATURES (RFECV)
# ----------------------------------------------------------
print("\n🤖 AI đang tự động dò tìm tập hợp đặc trưng tối ưu nhất...")
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

# Trích xuất danh sách các cột được AI giữ lại
features = list(np.array(all_possible_features)[selector.support_])

print(f"🔥 AI đã lọc từ {len(all_possible_features)} xuống còn "
      f"{len(features)} đặc trưng xuất sắc nhất!")
print(f"📋 Danh sách đặc trưng được chọn tự động: {features}\n")

# Chuẩn hóa dữ liệu dựa trên danh sách tính năng đã được AI chọn lọc
scaler = StandardScaler()
scaler.fit(train_data[features])
X_train_pool_scaled = scaler.transform(train_pool[features])

# ==========================================================
# 2. HUẤN LUYỆN 2 MÔ HÌNH (Dùng Features tối ưu mới)
# ==========================================================
print("🧠 ĐANG HUẤN LUYỆN MÔ HÌNH 1: VOTING ENSEMBLE...")
vote_top4, vote_top2, vote_champ = train_voting_baseline(
    X_train_pool_scaled, train_pool['Target_Top4'],
    train_pool['Target_Top2'], train_pool['Target_Champ']
)

print("🧠 ĐANG HUẤN LUYỆN MÔ HÌNH 2: STACKING TIÊU CHUẨN...")
stack_top4, stack_top2, stack_champ = train_stacking_baseline(
    X_train_pool_scaled, train_pool['Target_Top4'],
    train_pool['Target_Top2'], train_pool['Target_Champ']
)


# ==========================================================
# 3. ĐÁNH GIÁ MÔ HÌNH
# ==========================================================
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


print("⚙️ ĐANG CHẠY GIẢ LẬP MONTE CARLO CHO MODEL 1...")
v_top4, v_top2, v_champ = evaluate_model(vote_top4, vote_top2, vote_champ)

print("⚙️ ĐANG CHẠY GIẢ LẬP MONTE CARLO CHO MODEL 2...")
s_top4, s_top2, s_champ = evaluate_model(stack_top4, stack_top2, stack_champ)

print("\n" + "="*70)
print("📊 BẢNG SO SÁNH 2 MÔ HÌNH: VOTING vs STACKING 📊")
print("="*70)
print(f"| {'Tiêu chí Đánh giá':<20} | {'Mô hình 1 (Voting)':<20} | "
      f"{'Mô hình 2 (Stacking)':<20} |")
print("-" * 70)
print(f"| {'Dự đoán Top 4':<20} | {v_top4:>19.1f}% | {s_top4:>19.1f}% |")
print(f"| {'Dự đoán NBA Finals':<20} | {v_top2:>19.1f}% | {s_top2:>19.1f}% |")
print(f"| {'Đoán Nhà Vô Địch':<20} | {v_champ:>19.1f}% | {s_champ:>19.1f}% |")
print("="*70)

# ==========================================================
# 4. VẼ BIỂU ĐỒ VÀ LƯU RA THƯ MỤC 'RESULTS'
# ==========================================================
print("\n📈 Đang tạo các biểu đồ phân tích...")

labels = ['Top 4', 'Top 2 (Finals)', 'Nhà Vô Địch']
voting_scores = [v_top4, v_top2, v_champ]
stacking_scores = [s_top4, s_top2, s_champ]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width/2, voting_scores, width, label='Voting', color='#4c72b0')
plt.bar(x + width/2, stacking_scores, width, label='Stacking', color='#dd8452')

plt.ylabel('Độ chính xác (%)', fontweight='bold')
plt.title('SO SÁNH HIỆU SUẤT: VOTING VS STACKING', fontweight='bold')
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

# Vẽ biểu đồ Feature tầm quan trọng dựa trên tập đã chọn tự động
rf_estimator = vote_champ.named_estimators_['rf']
importances = rf_estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('MỨC ĐỘ QUAN TRỌNG CỦA CÁC TÍNH NĂNG DO AI CHỌN', fontweight='bold')
plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Mức độ ảnh hưởng (Relative Importance)')
plt.tight_layout()
plt.savefig('results/top15_feature_importance.png', dpi=300)
plt.close()

print("💾 Đã lưu 2 biểu đồ vào thư mục 'results/':")
print("   1. results/model_performance_comparison.png")
print("   2. results/top15_feature_importance.png")