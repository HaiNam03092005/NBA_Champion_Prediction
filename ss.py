import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
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
    'Playoff_Core_VORP_Share', 'Core_Fatigue_MP', 'Bench_Tactical_BPM'
]
features = [col for col in df.columns if col.endswith('_Rank') or col in absolute_features]

train_data = df[(df['Season_Year'] >= 2000) & (df['Season_Year'] <= 2020)].copy()
test_data  = df[(df['Season_Year'] >= 2021) & (df['Season_Year'] <= 2025)].copy()

scaler = StandardScaler()
scaler.fit(train_data[features])

# ==========================================================
# 2. XÂY DỰNG 4 POOL HUẤN LUYỆN PHÂN CẤP (HIERARCHICAL)
# ==========================================================
print("\n📦 Đang xây dựng 4 Pool huấn luyện phân cấp (Hierarchical)...")

pool_playoff = train_data.copy()
pool_top4    = train_data[train_data['Is_Playoff'] == 1].copy()
pool_top2    = train_data[train_data['Target_Top4'] == 1].copy()
pool_champ   = train_data[train_data['Target_Top2'] == 1].copy()

print(f"   Stage 1 (Playoff) : {len(pool_playoff)} mẫu")
print(f"   Stage 2 (Top 4)   : {len(pool_top4)} mẫu")
print(f"   Stage 3 (Top 2)   : {len(pool_top2)} mẫu")
print(f"   Stage 4 (Vô địch) : {len(pool_champ)} mẫu")


# ==========================================================
# 3. TẠO TimeSeriesSplit THEO MÙA GIẢI CHO GridSearchCV
# ==========================================================
# ┌──────────────────────────────────────────────────────────────────┐
# │  Tại sao TimeSeriesSplit chỉ dùng cho GridSearchCV?              │
# │                                                                  │
# │  GridSearchCV: chấp nhận list of (train_idx, val_idx) tùy ý     │
# │    → Dùng TimeSeriesSplit mùa giải, không xáo trộn              │
# │                                                                  │
# │  StackingClassifier: dùng cross_val_predict nội bộ, yêu cầu     │
# │  "partition" — TỪNG sample phải xuất hiện trong val đúng 1 lần  │
# │  TimeSeriesSplit expanding window vi phạm điều này vì các mùa   │
# │  đầu (2000-2004) không bao giờ được validate → lỗi ValueError   │
# │    → StackingClassifier dùng cv=3 cố định (integer), an toàn    │
# └──────────────────────────────────────────────────────────────────┘

def make_season_cv(pool_df, n_splits=4):
    """
    Tạo list CV folds theo mùa giải cho GridSearchCV.

    TimeSeriesSplit expanding window:
      Fold 1: Train [2000-2004] → Val [2005-2008]
      Fold 2: Train [2000-2008] → Val [2009-2012]
      Fold 3: Train [2000-2012] → Val [2013-2016]
      Fold 4: Train [2000-2016] → Val [2017-2020]

    Trả về: list of (train_indices, val_indices) theo row index của pool_df,
            hoặc integer 3 nếu pool quá nhỏ.
    """
    seasons = sorted(pool_df['Season_Year'].unique())
    n = len(seasons)
    safe_splits = min(n_splits, max(2, n // 3))
    tscv = TimeSeriesSplit(n_splits=safe_splits)
    season_arr = np.array(seasons)

    cv_folds = []
    for train_season_idx, val_season_idx in tscv.split(season_arr):
        train_seasons = set(season_arr[train_season_idx])
        val_seasons   = set(season_arr[val_season_idx])

        train_idx = np.where(pool_df['Season_Year'].isin(train_seasons))[0]
        val_idx   = np.where(pool_df['Season_Year'].isin(val_seasons))[0]

        if len(train_idx) >= 4 and len(val_idx) >= 2:
            cv_folds.append((train_idx, val_idx))

    return cv_folds if cv_folds else 3


# ==========================================================
# 4. HÀM TẠO SIÊU MÔ HÌNH
# ==========================================================
def tune_and_create_ensemble(pool_df, X, y, ai_name="AI", ensemble_method="voting"):
    """
    Kết hợp Hierarchical (Hướng 6) + TimeSeriesSplit CV (Hướng 2):
      - GridSearchCV   : cv = TimeSeriesSplit theo mùa giải (không xáo trộn)
      - Stacking final : cv = 3 cố định (bắt buộc vì cross_val_predict cần partition)
    """
    print(f"   ⚙️  Đang tối ưu {ai_name} ({ensemble_method.upper()}) với TimeSeriesCV...")

    cv_for_grid = make_season_cv(pool_df)
    n_folds = len(cv_for_grid) if isinstance(cv_for_grid, list) else cv_for_grid
    print(f"         → GridSearchCV: {n_folds} fold theo mùa | StackingCV: 3 (partition cố định)")

    # --- GridSearchCV: tìm hyperparameters tốt nhất theo thứ tự thời gian ---
    rf_params  = {'n_estimators': [100, 200], 'max_depth': [5, 8]}
    rf_grid    = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        rf_params, cv=cv_for_grid, n_jobs=-1)
    rf_grid.fit(X, y)

    svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    svm_grid   = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
        svm_params, cv=cv_for_grid, n_jobs=-1)
    svm_grid.fit(X, y)

    xgb_params = {'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
    xgb_grid   = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        xgb_params, cv=cv_for_grid, n_jobs=-1)
    xgb_grid.fit(X, y)

    best_estimators = [
        ('rf',  rf_grid.best_estimator_),
        ('svm', svm_grid.best_estimator_),
        ('xgb', xgb_grid.best_estimator_)
    ]

    # --- Ensemble: VotingClassifier hoặc StackingClassifier ---
    if ensemble_method == "voting":
        model = VotingClassifier(estimators=best_estimators, voting='soft')
    else:
        # cv=3 cố định cho StackingClassifier — cross_val_predict cần partition
        # (TimeSeriesSplit đã được áp dụng ở bước GridSearchCV ở trên rồi)
        model = StackingClassifier(
            estimators=best_estimators,
            final_estimator=LogisticRegression(class_weight='balanced'),
            cv=3)

    model.fit(X, y)
    return model


# ==========================================================
# 5. GÁN HẠT GIỐNG NBA ĐÚNG CHUẨN
# ==========================================================
def assign_seeds(conf_df):
    """
    Sắp xếp 8 đội theo W (desc), tiebreaker NRtg (desc).
    Seed 1-6: vào thẳng Playoffs.
    Seed 7-8: đã qua Play-In.
    """
    sorted_df = conf_df.sort_values(by=['W', 'NRtg'], ascending=[False, False]).reset_index(drop=True)
    sorted_df['Seed'] = range(1, len(sorted_df) + 1)
    return sorted_df


# ==========================================================
# 6. MÔ PHỎNG NHÁNH ĐẤU PHÂN CẤP
# ==========================================================
def simulate_conference_bracket(conf_df):
    """
    3 vòng đấu loại trực tiếp trong 1 Conference:
      R1 (First Round)      : 1v8, 4v5, 2v7, 3v6  → dùng Prob_Top4
      R2 (Semifinal)        : W(1v8) vs W(4v5),
                              W(2v7) vs W(3v6)     → dùng Prob_Top4
      R3 (Conference Finals): 2 người sót lại      → dùng Prob_Top2
    """
    teams = {int(row['Seed']): row for _, row in conf_df.iterrows()}

    def winner(t_a, t_b, prob_col):
        return t_a if t_a[prob_col] >= t_b[prob_col] else t_b

    w1 = winner(teams[1], teams[8], 'Prob_Top4')
    w2 = winner(teams[4], teams[5], 'Prob_Top4')
    w3 = winner(teams[2], teams[7], 'Prob_Top4')
    w4 = winner(teams[3], teams[6], 'Prob_Top4')

    finalist_a = winner(w1, w2, 'Prob_Top4')
    finalist_b = winner(w3, w4, 'Prob_Top4')

    conf_champion = winner(finalist_a, finalist_b, 'Prob_Top2')

    return [finalist_a['TEAM_NAME'], finalist_b['TEAM_NAME']], conf_champion


# ==========================================================
# 7. PIPELINE CHÍNH
# ==========================================================
def run_evaluation_pipeline(method="voting"):
    print(f"\n{'='*80}")
    print(f"🚀 HIERARCHICAL + TimeSeriesCV PIPELINE: {method.upper()} 🚀")
    print(f"{'='*80}")

    print("\n🧠 GIAI ĐOẠN HUẤN LUYỆN (4 TẦNG PHÂN CẤP):")

    X1 = scaler.transform(pool_playoff[features])
    ai_t1 = tune_and_create_ensemble(pool_playoff, X1, pool_playoff['Is_Playoff'],
                                      "Tầng1 (Playoff)", method)

    X2 = scaler.transform(pool_top4[features])
    ai_t2 = tune_and_create_ensemble(pool_top4, X2, pool_top4['Target_Top4'],
                                      "Tầng2 (Top4)", method)

    X3 = scaler.transform(pool_top2[features])
    ai_t3 = tune_and_create_ensemble(pool_top2, X3, pool_top2['Target_Top2'],
                                      "Tầng3 (Top2)", method)

    X4 = scaler.transform(pool_champ[features])
    ai_t4 = tune_and_create_ensemble(pool_champ, X4, pool_champ['Target_Champ'],
                                      "Tầng4 (Champ)", method)

    test_years = sorted(test_data['Season_Year'].unique())
    total_top4_correct = total_top2_correct = total_champ_correct = 0
    max_top4 = max_top2 = max_champ = 0

    print("\n🌲 KẾT QUẢ MÔ PHỎNG NHÁNH ĐẤU:")

    for year in test_years:
        print(f"\n📅 MÙA GIẢI {year}:")
        year_data = test_data[test_data['Season_Year'] == year].copy()

        actual_top4  = year_data[year_data['Target_Top4'] == 1]['TEAM_NAME'].tolist()
        actual_top2  = year_data[year_data['Target_Top2'] == 1]['TEAM_NAME'].tolist()
        actual_champ_row = year_data[year_data['Target_Champ'] == 1]
        actual_champ = actual_champ_row.iloc[0]['TEAM_NAME'] if not actual_champ_row.empty else "UNKNOWN"

        playoff_teams = year_data[year_data['Is_Playoff'] == 1].copy()
        east_df = playoff_teams[playoff_teams['Conference'] == 'East'].copy()
        west_df = playoff_teams[playoff_teams['Conference'] == 'West'].copy()

        if len(east_df) != 8 or len(west_df) != 8:
            print(f"   ⚠️  Không đủ 16 đội Playoff — bỏ qua.")
            continue

        east_seeded = assign_seeds(east_df)
        west_seeded = assign_seeds(west_df)
        playoff_16  = pd.concat([east_seeded, west_seeded]).reset_index(drop=True)

        X_test = scaler.transform(playoff_16[features])
        playoff_16['Prob_Top4']  = ai_t2.predict_proba(X_test)[:, 1]
        playoff_16['Prob_Top2']  = ai_t3.predict_proba(X_test)[:, 1]
        playoff_16['Prob_Champ'] = ai_t4.predict_proba(X_test)[:, 1]

        east_bracket = playoff_16[playoff_16['Conference'] == 'East']
        west_bracket = playoff_16[playoff_16['Conference'] == 'West']

        east_top4, east_champ = simulate_conference_bracket(east_bracket)
        west_top4, west_champ = simulate_conference_bracket(west_bracket)

        predicted_top4  = east_top4 + west_top4
        predicted_top2  = [east_champ['TEAM_NAME'], west_champ['TEAM_NAME']]
        predicted_champ = (east_champ['TEAM_NAME']
                           if east_champ['Prob_Champ'] >= west_champ['Prob_Champ']
                           else west_champ['TEAM_NAME'])

        if actual_top4:
            c = len(set(predicted_top4) & set(actual_top4))
            total_top4_correct += c; max_top4 += 4
            print(f"   🌪️  Top 4  (Trúng {c}/4): {', '.join(predicted_top4)}")
            print(f"              Thực tế : {', '.join(actual_top4)}")

        if actual_top2:
            c = len(set(predicted_top2) & set(actual_top2))
            total_top2_correct += c; max_top2 += 2
            print(f"   ⚔️  Finals (Trúng {c}/2): {', '.join(predicted_top2)}")
            print(f"              Thực tế : {', '.join(actual_top2)}")

        if actual_champ != "UNKNOWN":
            c = int(predicted_champ == actual_champ)
            total_champ_correct += c; max_champ += 1
            status = "✅ ĐÚNG" if c else "❌ SAI"
            print(f"   👑 Vô địch : {predicted_champ} | Thực tế: {actual_champ} ({status})")

    acc_top4  = (total_top4_correct / max_top4)  * 100 if max_top4  > 0 else 0
    acc_top2  = (total_top2_correct / max_top2)  * 100 if max_top2  > 0 else 0
    acc_champ = (total_champ_correct / max_champ) * 100 if max_champ > 0 else 0
    acc_overall = ((total_top4_correct + total_top2_correct + total_champ_correct) /
                   (max_top4 + max_top2 + max_champ)) * 100 if (max_top4 + max_top2 + max_champ) > 0 else 0

    return acc_top4, acc_top2, acc_champ, acc_overall


# ==========================================================
# 8. THỰC THI VÀ SO SÁNH
# ==========================================================
print("\n💡 KHỞI CHẠY: HIERARCHICAL TARGET + TimeSeriesSplit CV...")

v_top4, v_top2, v_champ, v_overall = run_evaluation_pipeline(method="voting")
s_top4, s_top2, s_champ, s_overall = run_evaluation_pipeline(method="stacking")

print("\n" + "="*80)
print("📊 BẢNG SO SÁNH: HIERARCHICAL + TimeSeriesSplit CV 📊")
print("="*80)
print(f"| {'Tiêu chí':<22} | {'VOTING':<22} | {'STACKING':<22} |")
print("-" * 74)
print(f"| {'Chọn Top 4':<22} | {v_top4:>19.1f}% | {s_top4:>19.1f}% |")
print(f"| {'Chọn Top 2 (Finals)':<22} | {v_top2:>19.1f}% | {s_top2:>19.1f}% |")
print(f"| {'Đoán Nhà Vô Địch':<22} | {v_champ:>19.1f}% | {s_champ:>19.1f}% |")
print("-" * 74)
print(f"| {'TỔNG QUAN (OVERALL)':<22} | {v_overall:>19.1f}% | {s_overall:>19.1f}% |")
print("="*80)
print("""
📌 Kiến trúc kết hợp Hướng 6 + Hướng 2:
   ┌─ Hierarchical (Hướng 6) ──────────────────────────────────────┐
   │  Tầng 1: 630 mẫu  (30 đội/mùa)  → Is_Playoff               │
   │  Tầng 2: 336 mẫu  (16 đội/mùa)  → Target_Top4              │
   │  Tầng 3:  84 mẫu  (4 đội/mùa)   → Target_Top2              │
   │  Tầng 4:  42 mẫu  (2 đội/mùa)   → Target_Champ             │
   └───────────────────────────────────────────────────────────────┘
   ┌─ TimeSeriesSplit CV (Hướng 2) ────────────────────────────────┐
   │  GridSearchCV dùng CV theo mùa (không xáo trộn):            │
   │    Fold 1: Train [2000-2004] → Val [2005-2008]              │
   │    Fold 2: Train [2000-2008] → Val [2009-2012]              │
   │    Fold 3: Train [2000-2012] → Val [2013-2016]              │
   │    Fold 4: Train [2000-2016] → Val [2017-2020]              │
   │  StackingClassifier dùng cv=3 (bắt buộc vì partition rule) │
   └───────────────────────────────────────────────────────────────┘
""")