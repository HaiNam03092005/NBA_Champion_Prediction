import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN FILE
# ==========================================
# File này đọc trực tiếp kết quả đã được gộp hoàn hảo từ merge_all.py
INPUT_FILE = 'data/processed/final_complete_dataset.csv'
OUTPUT_FILE = 'data/processed/model_ready_dataset.csv'

print("============ TIẾN TRÌNH TẠO ĐẶC TRƯNG BẬC CAO (ADVANCED FEATURES) ============")

# ==========================================
# 2. ĐỌC DỮ LIỆU GỐC
# ==========================================
print(f"🔄 Bước 1: Đang đọc dữ liệu từ '{INPUT_FILE}'...")
if not os.path.exists(INPUT_FILE):
    print(f"❌ LỖI: Không tìm thấy file '{INPUT_FILE}'. Vui lòng chạy merge_all.py trước!")
    exit()

df = pd.read_csv(INPUT_FILE)

# ==========================================
# 3. [BỔ SUNG TỪ LUKE-LITE] XỬ LÝ CHỈ SỐ ELO RATING
# ==========================================
print("💡 Bước 1.5: Kiểm tra và tối ưu Proxy Elo Rating...")
if 'Elo_Score' not in df.columns and all(c in df.columns for c in ['W', 'L', 'SRS']):
    win_pct = df['W'] / (df['W'] + df['L']).replace(0, 0.5)
    df['Elo_Score'] = 1500 + (win_pct - 0.5) * 300 + df['SRS'] * 15
elif 'Elo_Score' in df.columns:
    print("   📈 Đã có sẵn chỉ số 'Elo_Score' trong dữ liệu.")

# ==========================================
# 4. TÍNH TOÁN ĐỘ TUỔI HOÀNG KIM (AGE DIFF)
# ==========================================
print("🧬 Bước 2: Bổ sung chỉ số chênh lệch Tuổi hoàng kim (Age_Diff)...")
if 'Age' in df.columns and 'Target_Champ' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    champ_mean_age = df[df['Target_Champ'] == 1]['Age'].mean()
    if pd.isna(champ_mean_age): champ_mean_age = 28.0 
    
    df['Age_Diff'] = abs(df['Age'] - champ_mean_age)
    df['Age_Diff_Rank'] = df.groupby('Season_Year')['Age_Diff'].rank(ascending=True, method='min')

# ==========================================
# 5. TÍNH TOÁN CÁC CỘT XẾP HẠNG CƠ BẢN & ĐỐI KHÁNG
# ==========================================
print("📊 Bước 3: Đang tính toán thứ hạng (Rank) cốt lõi, Thể lực & Đặc trưng Đối kháng...")
rank_metrics = {
    # --- Các chỉ số Đội bóng cơ bản ---
    'W': False, 'SRS': False, 'NRtg': False, 'ORtg': False, 
    'DRtg': True, 'MOV': False, 'TS%': False,
    'Team_Total_VORP': False, 'Team_Avg_OBPM': False, 'Team_Avg_DBPM': False,
    
    # --- Chỉ số thể lực & Đội hình ---
    'Playoff_Core_VORP_Share': False,  
    'Bench_Tactical_BPM': False,       
    'Core_Fatigue_MP': True,           
    
    # --- Đặc trưng phong độ nâng cao ---
    'Elo_Score': False,
    
    # --- THỨ HẠNG CỦA CÁC ĐẶC TRƯNG ĐỐI KHÁNG (Đã được merge_all gộp vào) ---
    'Diff_PTS': False,   # Hiệu số điểm càng cao -> Hạng càng tốt (Hạng 1)
    'Diff_eFG': False,   # Hiệu số hiệu suất rổ càng cao -> Hạng càng tốt
    'Diff_REB': False,   # Thắng kiểm soát bóng càng nhiều -> Hạng càng tốt
    'Diff_TOV': True     # Lỗi mất bóng ít hơn đối thủ (Chênh lệch Âm) -> Hạng càng tốt (True)
}

for col, is_ascending in rank_metrics.items():
    if col in df.columns:
        df[f"{col}_Rank"] = df.groupby('Season_Year')[col].rank(ascending=is_ascending, method='min')

# ==========================================
# 6. CÁC LUẬT BÓNG RỔ THỰC TẾ (DOMAIN KNOWLEDGE)
# ==========================================
print("🧠 Bước 4: Khởi tạo biến Domain Knowledge (Elite Defense, Two-Way, eFG_Diff, Superstar)...")

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

# ==========================================
# 7. DBSCAN PHÂN CỤM VƯƠNG TRIỀU (DYNASTY CLUSTERS)
# ==========================================
print("🔮 Bước 4.5: Áp dụng thuật toán mật độ DBSCAN tìm Dynasty Profile...")

features_for_clustering = ['SRS', 'NRtg', 'TS%', 'Superstar_Impact_Score']
existing_clust_features = [c for c in features_for_clustering if c in df.columns]

if len(existing_clust_features) == len(features_for_clustering):
    X_cluster = df[existing_clust_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X_cluster)

    dbscan = DBSCAN(eps=0.6, min_samples=3)
    df['Dynasty_Cluster'] = dbscan.fit_predict(X_scaled)

    df['Is_Dynasty_Profile'] = (df['Dynasty_Cluster'] != -1).astype(int)
    print(f"   📊 Số lượng đội bóng lọt vào profile Vương Triều: {df['Is_Dynasty_Profile'].sum()} đội.")
else:
    print("   ⚠️ CẢNH BÁO: Không thể chạy DBSCAN do thiếu một trong các trường đặc trưng cốt lõi.")

# ==========================================
# 8. LƯU DỮ LIỆU SẴN SÀNG CHO MODEL
# ==========================================
print(f"\n💾 Bước 5: Đang đóng gói dữ liệu siêu việt ra file '{OUTPUT_FILE}'...")
if 'Conf_Seed' in df.columns:
    df = df.sort_values(by=['Season_Year', 'Conference', 'Conf_Seed'], ascending=[False, True, True])

df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Đã thêm Xếp hạng (Rank) cho chỉ số Elo Rating, Thể lực & Chiều sâu đội hình.")
print(f"✅ Đã tích hợp thành công bộ lọc phân cụm DBSCAN Dynasty Classifier.")
print(f"✅ Kích thước Dataset sẵn sàng huấn luyện: {df.shape[0]} dòng, {df.shape[1]} cột.")