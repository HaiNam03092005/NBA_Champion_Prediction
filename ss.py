import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# ==========================================================
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================================
print("📂 Đang tải dữ liệu từ 'data/processed/model_ready_dataset.csv'...")
df = pd.read_csv('data/processed/model_ready_dataset.csv')
df.fillna(0, inplace=True)

absolute_features = [
    'Has_Top5_MVP', 'All_NBA_Count', 'Prev_Year_Playoff_Round',
    'Age_Diff', 'Elite_Defense_Flag', 'Two_Way_Score', 'eFG_Diff', 'Superstar_Impact_Score',
    'Playoff_Core_VORP_Share', 'Core_Fatigue_MP', 'Bench_Tactical_BPM', 'Is_Dynasty_Profile'
]
features = [col for col in df.columns if col.endswith('_Rank') or col in absolute_features]

train_data = df[(df['Season_Year'] >= 2000) & (df['Season_Year'] <= 2020)].copy()
test_data = df[(df['Season_Year'] >= 2021) & (df['Season_Year'] <= 2025)].copy()

scaler = StandardScaler()
scaler.fit(train_data[features])

train_pool = train_data[train_data['Is_Playoff'] == 1].copy()

## ==========================================================
# 2. HÀM TẠO CHUỖI MÔ HÌNH PHÂN CẤP (HIERARCHICAL STACKING)
# ==========================================================
def train_hierarchical_models(train_df, feature_cols):
    print("\n🧠 ĐANG XÂY DỰNG CHUỖI MÔ HÌNH PHÂN CẤP (NATURE 2025 ARCHITECTURE)...")
    
    cv_setting = 5
    
    base_learners = [
        ('xgb', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42, eval_metric='logloss')),
        ('lgb', LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42, verbose=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42))
    ]
    
    # Thiết lập 2 Meta-Learner riêng biệt để tối ưu bài toán mất cân bằng dữ liệu
    meta_learner_standard = LogisticRegression() # Dành cho Top 4 và Top 2
    meta_learner_champ = LogisticRegression(class_weight='balanced', C=0.5) # Ép tính tự tin cho Tầng Vô địch
    
    X_base = scaler.transform(train_df[feature_cols])
    
    # --- TẦNG 1: HUẤN LUYỆN AI_TOP4 ---
    print("   ⚙️ Tầng 1: Đang huấn luyện AI dự đoán Top 4...")
    y_top4 = train_df['Target_Top4']
    model_top4 = StackingClassifier(estimators=base_learners, final_estimator=meta_learner_standard, cv=cv_setting)
    model_top4.fit(X_base, y_top4)
    prob_top4_feature = model_top4.predict_proba(X_base)[:, 1]
    
    # --- TẦNG 2: HUẤN LUYỆN AI_TOP2 ---
    print("   ⚙️ Tầng 2: Đang huấn luyện AI dự đoán Top 2...")
    X_stage2 = np.column_stack((X_base, prob_top4_feature)) 
    y_top2 = train_df['Target_Top2']
    model_top2 = StackingClassifier(estimators=base_learners, final_estimator=meta_learner_standard, cv=cv_setting)
    model_top2.fit(X_stage2, y_top2)
    prob_top2_feature = model_top2.predict_proba(X_stage2)[:, 1]
    
    # --- TẦNG 3: HUẤN LUYỆN AI_CHAMPION ---
    print("   ⚙️ Tầng 3: Đang huấn luyện AI dự đoán Vô Địch...")
    X_stage3 = np.column_stack((X_stage2, prob_top2_feature)) 
    y_champ = train_df['Target_Champ']
    
    # Sử dụng meta_learner_champ (balanced) để giải quyết tỷ lệ 1/15 cực đoan của nhà vô địch
    model_champ = StackingClassifier(estimators=base_learners, final_estimator=meta_learner_champ, cv=cv_setting)
    model_champ.fit(X_stage3, y_champ)
    
    return model_top4, model_top2, model_champ

# ==========================================================
# 3. THUẬT TOÁN GIẢ LẬP NHÁNH ĐẤU (DETERMINISTIC LOG-5)
# ==========================================================
def log5_matchup_probability(prob_a, prob_b):
    numerator = prob_a * (1.0 - prob_b)
    denominator = prob_a * (1.0 - prob_b) + prob_b * (1.0 - prob_a)
    if denominator == 0: return 0.5
    return numerator / denominator

def simulate_best_of_7_series(team_a, team_b, prob_col):
    """Xác suất đối đầu (Log-5) đội nào >= 50% thì chốt cứng đội đó thắng"""
    p_a_wins_single_game = log5_matchup_probability(team_a[prob_col], team_b[prob_col])
    if p_a_wins_single_game >= 0.5:
        return team_a
    else:
        return team_b

def simulate_conference_bracket(conf_df):
    """Mô phỏng cây nhánh Playoff dùng thuật toán Bo7"""
    teams = {row['Seed']: row for _, row in conf_df.iterrows()}
    
    # --- VÒNG 1: TỨ KẾT MIỀN ---
    w1_8 = simulate_best_of_7_series(teams[1], teams[8], 'Prob_Top4')
    w4_5 = simulate_best_of_7_series(teams[4], teams[5], 'Prob_Top4')
    w2_7 = simulate_best_of_7_series(teams[2], teams[7], 'Prob_Top4')
    w3_6 = simulate_best_of_7_series(teams[3], teams[6], 'Prob_Top4')
    
    # --- VÒNG 2: BÁN KẾT MIỀN ---
    w_top4_a = simulate_best_of_7_series(w1_8, w4_5, 'Prob_Top4')
    w_top4_b = simulate_best_of_7_series(w2_7, w3_6, 'Prob_Top4')
    conf_predicted_top4 = [w_top4_a['TEAM_NAME'], w_top4_b['TEAM_NAME']]
    
    # --- VÒNG 3: CHUNG KẾT MIỀN (Xác định vé vào Chung kết tổng) ---
    conf_champion = simulate_best_of_7_series(w_top4_a, w_top4_b, 'Prob_Top2')
    
    return conf_predicted_top4, conf_champion

# ==========================================================
# 4. HÀM PIPELINE KIỂM TRA TRÊN THỂ THỨC CÂY NHÁNH CỐ ĐỊNH
# ==========================================================
def run_evaluation_pipeline():
    print(f"\n" + "="*80)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN VÀ MÔ PHỎNG PLAYOFFS (HIERARCHICAL + DETERMINISTIC) 🚀")
    print("="*80)
    
    ai_top4, ai_top2, ai_champ = train_hierarchical_models(train_pool, features)

    test_years = sorted(test_data['Season_Year'].unique())
    total_top4_correct, total_top2_correct, total_champ_correct = 0, 0, 0
    max_top4, max_top2, max_champ = 0, 0, 0

    print("\n🌲 KẾT QUẢ TEST MÔ PHỎNG THEO ĐÚNG SƠ ĐỒ CÂY PLAYOFF NBA 🌲")
    
    for year in test_years:
        print(f"\n📅 MÙA GIẢI {year}:")
        year_data = test_data[test_data['Season_Year'] == year].copy()
        
        actual_top4 = year_data[year_data['Target_Top4'] == 1]['TEAM_NAME'].tolist()
        actual_top2 = year_data[year_data['Target_Top2'] == 1]['TEAM_NAME'].tolist()
        actual_champ_row = year_data[year_data['Target_Champ'] == 1]
        actual_champ = actual_champ_row.iloc[0]['TEAM_NAME'] if not actual_champ_row.empty else "UNKNOWN"
        
        actual_playoff_teams = year_data[year_data['Is_Playoff'] == 1].copy()
        east_playoffs = actual_playoff_teams[actual_playoff_teams['Conference'] == 'East'].copy()
        west_playoffs = actual_playoff_teams[actual_playoff_teams['Conference'] == 'West'].copy()
        
        if len(east_playoffs) != 8 or len(west_playoffs) != 8:
            print(f"   ⚠️ Dữ liệu năm {year} không đủ chuẩn 16 đội Playoff, bỏ qua.")
            continue
            
        east_playoffs = east_playoffs.sort_values(by=['W', 'NRtg'], ascending=[False, False])
        west_playoffs = west_playoffs.sort_values(by=['W', 'NRtg'], ascending=[False, False])
        east_playoffs['Seed'] = range(1, 9)
        west_playoffs['Seed'] = range(1, 9)
        
        playoff_16_teams = pd.concat([east_playoffs, west_playoffs])
        X_test_base = scaler.transform(playoff_16_teams[features])
        
        playoff_16_teams['Prob_Top4'] = ai_top4.predict_proba(X_test_base)[:, 1]
        
        X_test_stage2 = np.column_stack((X_test_base, playoff_16_teams['Prob_Top4']))
        playoff_16_teams['Prob_Top2'] = ai_top2.predict_proba(X_test_stage2)[:, 1]
        
        X_test_stage3 = np.column_stack((X_test_stage2, playoff_16_teams['Prob_Top2']))
        playoff_16_teams['Prob_Champ'] = ai_champ.predict_proba(X_test_stage3)[:, 1]
        
        east_finalists = playoff_16_teams[playoff_16_teams['Conference'] == 'East']
        west_finalists = playoff_16_teams[playoff_16_teams['Conference'] == 'West']
        
        east_top4, east_champ = simulate_conference_bracket(east_finalists)
        west_top4, west_champ = simulate_conference_bracket(west_finalists)
        
        predicted_top4 = east_top4 + west_top4
        predicted_top2 = [east_champ['TEAM_NAME'], west_champ['TEAM_NAME']]
        
        predicted_champ_row = simulate_best_of_7_series(east_champ, west_champ, 'Prob_Champ')
        predicted_champ = predicted_champ_row['TEAM_NAME']
            
        if len(actual_top4) > 0:
            correct_in_top4 = len(set(predicted_top4) & set(actual_top4))
            total_top4_correct += correct_in_top4
            max_top4 += 4
            print(f"   🌪️ Dự đoán Top 4 (Trúng {correct_in_top4}/4): {', '.join(predicted_top4)}")
        
        if len(actual_top2) > 0:
            correct_in_top2 = len(set(predicted_top2) & set(actual_top2))
            total_top2_correct += correct_in_top2
            max_top2 += 2
            print(f"   ⚔️ Dự đoán NBA Finals (Trúng {correct_in_top2}/2): {', '.join(predicted_top2)}")

        if actual_champ != "UNKNOWN":
            is_champ_correct = 1 if predicted_champ == actual_champ else 0
            total_champ_correct += is_champ_correct
            max_champ += 1
            print(f"   👑 AI Chọn Vô Địch Tổng >> {predicted_champ} << | Thực tế: {actual_champ} " + ("(✅ ĐÚNG)" if is_champ_correct else "(❌ SAI)"))

    acc_top4 = (total_top4_correct / max_top4) * 100 if max_top4 > 0 else 0
    acc_top2 = (total_top2_correct / max_top2) * 100 if max_top2 > 0 else 0
    acc_champ = (total_champ_correct / max_champ) * 100 if max_champ > 0 else 0
    
    overall_correct = total_top4_correct + total_top2_correct + total_champ_correct
    max_overall = max_top4 + max_top2 + max_champ
    acc_overall = (overall_correct / max_overall) * 100 if max_overall > 0 else 0
    
    return acc_top4, acc_top2, acc_champ, acc_overall

# ==========================================================
# 5. THỰC THI
# ==========================================================
s_top4, s_top2, s_champ, s_overall = run_evaluation_pipeline()

print("\n" + "="*80)
print("📊 BẢNG KẾT QUẢ: MÔ HÌNH HIERARCHICAL STACKING & DETERMINISTIC PLAYOFF 📊")
print("="*80)
print(f"| {'Tiêu chí':<20} | {'Độ chính xác (Accuracy)':<30} |")
print("-" * 55)
print(f"| {'Chọn Top 4':<20} | {s_top4:>29.1f}% |")
print(f"| {'Chọn Top 2':<20} | {s_top2:>29.1f}% |")
print(f"| {'Đoán Nhà Vô Địch':<20} | {s_champ:>29.1f}% |")
print("-" * 55)
print(f"| {'TỔNG QUAN (OVERALL)':<20} | {s_overall:>29.1f}% |")
print("="*80)