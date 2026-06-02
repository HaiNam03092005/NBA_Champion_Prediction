import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================================
print("📂 Đang tải dữ liệu từ 'data/model_ready_dataset.csv'...")
df = pd.read_csv('data/model_ready_dataset.csv')
df.fillna(0, inplace=True)

absolute_features = ['Has_Top5_MVP', 'All_NBA_Count', 'Prev_Year_Playoff_Round']
features = [col for col in df.columns if col.endswith('_Rank') or col in absolute_features]

train_data = df[(df['Season_Year'] >= 2000) & (df['Season_Year'] <= 2020)].copy()
test_data = df[(df['Season_Year'] >= 2021) & (df['Season_Year'] <= 2025)].copy()

scaler = StandardScaler()
scaler.fit(train_data[features])

# ==========================================================
# 2. HÀM TẠO SIÊU MÔ HÌNH (VOTING HOẶC STACKING)
# ==========================================================
def tune_and_create_ensemble(X, y, ai_name="AI", ensemble_method="voting"):
    print(f"   ⚙️ Đang tối ưu thông số cho {ai_name} ({ensemble_method.upper()})...")
    
    # 1. Random Forest
    rf_params = {'n_estimators': [100], 'max_depth': [5, 8]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), rf_params, cv=3, n_jobs=-1)
    rf_grid.fit(X, y)
    
    # 2. SVM
    svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    svm_grid = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'), svm_params, cv=3, n_jobs=-1)
    svm_grid.fit(X, y)
    
    # 3. XGBoost
    xgb_params = {'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
    xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, cv=3, n_jobs=-1)
    xgb_grid.fit(X, y)
    
    best_estimators = [
        ('rf', rf_grid.best_estimator_), 
        ('svm', svm_grid.best_estimator_),
        ('xgb', xgb_grid.best_estimator_)
    ]
    
    # CHỌN PHƯƠNG PHÁP TỔNG HỢP (ENSEMBLE)
    if ensemble_method == "voting":
        model = VotingClassifier(estimators=best_estimators, voting='soft')
    else:  # stacking
        # Dùng Logistic Regression làm trọng tài Tầng 2 để ra quyết định cuối cùng
        model = StackingClassifier(estimators=best_estimators, final_estimator=LogisticRegression(), cv=3)
        
    model.fit(X, y)
    return model

# ==========================================================
# 3. HÀM CHẠY PIPELINE HUẤN LUYỆN & KIỂM TRA
# ==========================================================
def run_evaluation_pipeline(method="voting"):
    print(f"\n" + "="*80)
    print(f"🚀 BẮT ĐẦU CHUỖI HUẤN LUYỆN VÀ DỰ ĐOÁN BẰNG PHƯƠNG PHÁP: {method.upper()} 🚀")
    print("="*80)
    
    # --- HUẤN LUYỆN AI ---
    train_data_top10 = train_data[train_data['W_Rank'] <= 10].copy()
    X1_train, y1_train = scaler.transform(train_data_top10[features]), train_data_top10['Target_Top4']
    ai_1_top4 = tune_and_create_ensemble(X1_train, y1_train, "AI_1 (Top 4)", method)

    train_data_top4 = train_data[train_data['Target_Top4'] == 1].copy()
    X2_train, y2_train = scaler.transform(train_data_top4[features]), train_data_top4['Target_Top2']
    ai_2_top2 = tune_and_create_ensemble(X2_train, y2_train, "AI_2 (Top 2)", method)

    train_data_top2 = train_data[train_data['Target_Top2'] == 1].copy()
    X3_train, y3_train = scaler.transform(train_data_top2[features]), train_data_top2['Target_Champ']
    ai_3_champ = tune_and_create_ensemble(X3_train, y3_train, "AI_3 (Vô địch)", method)

    # --- KIỂM TRA TRÊN TẬP TEST ---
    test_years = sorted(test_data['Season_Year'].unique())
    total_top4_correct, total_top2_correct, total_champ_correct = 0, 0, 0
    max_top4, max_top2, max_champ = 0, 0, 0

    print("\n🏆 KẾT QUẢ TEST QUA HỆ THỐNG PHỄU CÓ ÉP LUẬT CONFERENCE 🏆")
    
    for year in test_years:
        print(f"\n📅 MÙA GIẢI {year}:")
        year_data = test_data[test_data['Season_Year'] == year].copy()
        
        actual_top4 = year_data[year_data['Target_Top4'] == 1]['TEAM_NAME'].tolist()
        actual_top2 = year_data[year_data['Target_Top2'] == 1]['TEAM_NAME'].tolist()
        actual_champ_row = year_data[year_data['Target_Champ'] == 1]
        actual_champ = actual_champ_row.iloc[0]['TEAM_NAME'] if not actual_champ_row.empty else "UNKNOWN"
        
        year_data_top10 = year_data[year_data['W_Rank'] <= 10].copy()
        if year_data_top10.empty:
            continue
            
        # --------------------------------------------------
        # VÒNG 1: DỰ ĐOÁN TOP 4 (Ép luật: 2 Đông, 2 Tây)
        # --------------------------------------------------
        X_top10_scaled = scaler.transform(year_data_top10[features])
        year_data_top10['Prob_Top4'] = ai_1_top4.predict_proba(X_top10_scaled)[:, 1]
        
        east_top10 = year_data_top10[year_data_top10['Conference'] == 'East']
        west_top10 = year_data_top10[year_data_top10['Conference'] == 'West']
        
        # Chọn 2 Đông, 2 Tây cao điểm nhất
        predicted_top4_df = pd.concat([east_top10.nlargest(2, 'Prob_Top4'), west_top10.nlargest(2, 'Prob_Top4')])
        predicted_top4 = predicted_top4_df['TEAM_NAME'].tolist()
        
        # --------------------------------------------------
        # VÒNG 2: DỰ ĐOÁN TOP 2 (Ép luật: 1 Đông, 1 Tây)
        # --------------------------------------------------
        X_top4_scaled = scaler.transform(predicted_top4_df[features])
        predicted_top4_df['Prob_Top2'] = ai_2_top2.predict_proba(X_top4_scaled)[:, 1]
        
        east_top4 = predicted_top4_df[predicted_top4_df['Conference'] == 'East']
        west_top4 = predicted_top4_df[predicted_top4_df['Conference'] == 'West']
        
        # Chọn 1 Đông, 1 Tây cao điểm nhất từ Top 4
        predicted_top2_df = pd.concat([east_top4.nlargest(1, 'Prob_Top2'), west_top4.nlargest(1, 'Prob_Top2')])
        predicted_top2 = predicted_top2_df['TEAM_NAME'].tolist()
        
        # --------------------------------------------------
        # VÒNG 3: DỰ ĐOÁN VÔ ĐỊCH
        # --------------------------------------------------
        X_top2_scaled = scaler.transform(predicted_top2_df[features])
        predicted_top2_df['Prob_Champ'] = ai_3_champ.predict_proba(X_top2_scaled)[:, 1]
        predicted_champ = predicted_top2_df.nlargest(1, 'Prob_Champ').iloc[0]['TEAM_NAME']
        
        # --- CHẤM ĐIỂM ---
        if len(actual_top4) > 0:
            correct_in_top4 = len(set(predicted_top4) & set(actual_top4))
            total_top4_correct += correct_in_top4
            max_top4 += 4
            print(f"   🌪️ Top 4 (Trúng {correct_in_top4}/4): {', '.join(predicted_top4)}")
        else:
            print(f"   🌪️ Top 4 (AI DỰ ĐOÁN): {', '.join(predicted_top4)} (Chưa có kết quả thực tế)")

        if len(actual_top2) > 0:
            correct_in_top2 = len(set(predicted_top2) & set(actual_top2))
            total_top2_correct += correct_in_top2
            max_top2 += 2
            print(f"   ⚔️ Top 2 (Trúng {correct_in_top2}/2): {', '.join(predicted_top2)}")
        else:
            print(f"   ⚔️ Top 2 (AI DỰ ĐOÁN): {', '.join(predicted_top2)} (Chưa có kết quả thực tế)")

        if actual_champ != "UNKNOWN":
            is_champ_correct = 1 if predicted_champ == actual_champ else 0
            total_champ_correct += is_champ_correct
            max_champ += 1
            print(f"   👑 Vô địch : AI chọn >> {predicted_champ} << | Thực tế: {actual_champ} " + ("(✅ ĐÚNG)" if is_champ_correct else "(❌ SAI)"))
        else:
            print(f"   👑 Vô địch : AI DỰ ĐOÁN >> {predicted_champ} << ⏳ (Chưa có kết quả thực tế, BỎ QUA chấm điểm)")

    # --- TỔNG KẾT TỈ LỆ ---
    acc_top4 = (total_top4_correct / max_top4) * 100 if max_top4 > 0 else 0
    acc_top2 = (total_top2_correct / max_top2) * 100 if max_top2 > 0 else 0
    acc_champ = (total_champ_correct / max_champ) * 100 if max_champ > 0 else 0
    
    overall_correct = total_top4_correct + total_top2_correct + total_champ_correct
    max_overall = max_top4 + max_top2 + max_champ
    acc_overall = (overall_correct / max_overall) * 100 if max_overall > 0 else 0
    
    return acc_top4, acc_top2, acc_champ, acc_overall

# ==========================================================
# 4. THỰC THI VÀ SO SÁNH (VOTING VS STACKING)
# ==========================================================
print("💡 HỆ THỐNG SẼ CHẠY CẢ 2 PHƯƠNG PHÁP ĐỂ TÌM RA THUẬT TOÁN ĐỈNH NHẤT...")

# Chạy Voting
v_top4, v_top2, v_champ, v_overall = run_evaluation_pipeline(method="voting")

# Chạy Stacking
s_top4, s_top2, s_champ, s_overall = run_evaluation_pipeline(method="stacking")

# Bảng so sánh chung cuộc
print("\n" + "="*80)
print("📊 BẢNG SO SÁNH CHUNG CUỘC: VOTING vs STACKING 📊")
print("="*80)
print(f"| {'Tiêu chí':<20} | {'VOTING (Biểu quyết)':<22} | {'STACKING (Xếp chồng)':<22} |")
print("-" * 72)
print(f"| {'Chọn Top 4':<20} | {v_top4:>19.1f}% | {s_top4:>19.1f}% |")
print(f"| {'Chọn Top 2':<20} | {v_top2:>19.1f}% | {s_top2:>19.1f}% |")
print(f"| {'Đoán Nhà Vô Địch':<20} | {v_champ:>19.1f}% | {s_champ:>19.1f}% |")
print("-" * 72)
print(f"| {'TỔNG QUAN (OVERALL)':<20} | {v_overall:>19.1f}% | {s_overall:>19.1f}% |")
print("="*80)