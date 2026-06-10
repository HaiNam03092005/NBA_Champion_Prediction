import pandas as pd
import numpy as np
import os

# ==========================================
# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN FILE
# ==========================================
INPUT_FILE = 'data/processed/final_complete_dataset.csv'
OUTPUT_FILE = 'data/processed/model_ready_dataset.csv'

print("============ TIẾN TRÌNH TẠO ĐẶC TRƯNG BẬC CAO (ADVANCED FEATURES) ============")

# ==========================================
# 2. ĐỌC DỮ LIỆU
# ==========================================
print(f"🔄 Bước 1: Đang đọc dữ liệu từ '{INPUT_FILE}'...")
if not os.path.exists(INPUT_FILE):
    print(f"❌ LỖI: Không tìm thấy file '{INPUT_FILE}'.")
    exit()

df = pd.read_csv(INPUT_FILE)

# ==========================================
# [BỔ SUNG TỪ LUKE-LITE] XỬ LÝ CHỈ SỐ ELO RATING
# ==========================================
if 'Elo_Score' not in df.columns and all(c in df.columns for c in ['W', 'L', 'SRS']):
    print("💡 Gợi ý từ luke-lite: Không tìm thấy 'Elo_Score' gốc, tự động tối ưu hóa Proxy Elo dựa trên Win% và SRS...")
    # Thuật toán xây dựng điểm Elo cơ sở (Điểm sàn 1500 + Biến động phong độ Regular Season)
    win_pct = df['W'] / (df['W'] + df['L']).replace(0, 0.5)
    df['Elo_Score'] = 1500 + (win_pct - 0.5) * 300 + df['SRS'] * 15
elif 'Elo_Score' in df.columns:
    print("📈 Tìm thấy chỉ số 'Elo_Score' từ hệ thống Scraper của luke-lite.")

# ==========================================
# 3. TÍNH TOÁN ĐỘ TUỔI HOÀNG KIM (AGE DIFF)
# ==========================================
print("🧬 Bước 2: Bổ sung chỉ số chênh lệch Tuổi hoàng kim (Age_Diff)...")
if 'Age' in df.columns and 'Target_Champ' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    champ_mean_age = df[df['Target_Champ'] == 1]['Age'].mean()
    if pd.isna(champ_mean_age): champ_mean_age = 28.0 
    
    df['Age_Diff'] = abs(df['Age'] - champ_mean_age)
    df['Age_Diff_Rank'] = df.groupby('Season_Year')['Age_Diff'].rank(ascending=True, method='min')

# ==========================================
# 4. TÍNH TOÁN CÁC CỘT XẾP HẠNG CƠ BẢN (RANKING)
# ==========================================
print("📊 Bước 3: Đang tính toán thứ hạng (Rank) cốt lõi và Thể lực...")
rank_metrics = {
    # --- Các chỉ số Đội bóng cơ bản (Team Stats) ---
    'W': False, 'SRS': False, 'NRtg': False, 'ORtg': False, 
    'DRtg': True, 'MOV': False, 'TS%': False,
    'Team_Total_VORP': False, 'Team_Avg_OBPM': False, 'Team_Avg_DBPM': False,
    
    # --- CHỈ SỐ THỂ LỰC & ĐỘI HÌNH ---
    'Playoff_Core_VORP_Share': False,  # Bộ khung 7 người càng gánh team -> Hạng càng cao
    'Bench_Tactical_BPM': False,       # Ghế dự bị càng xịn -> Hạng càng cao
    'Core_Fatigue_MP': True,           # Số phút cày ải THẤP (Ít mệt mỏi nhất) -> Hạng càng cao
    
    # --- ĐẶC TRƯNG PHONG ĐỘ NÂNG CAO (MỚI THÊM TỪ LUKE-LITE) ---
    'Elo_Score': False                 # Điểm số Elo càng cao -> Thứ hạng Elo_Rank càng lớn (hạng 1)
}

for col, is_ascending in rank_metrics.items():
    if col in df.columns:
        df[f"{col}_Rank"] = df.groupby('Season_Year')[col].rank(ascending=is_ascending, method='min')

# ==========================================
# 5. CÁC LUẬT BÓNG RỔ THỰC TẾ (DOMAIN KNOWLEDGE)
# ==========================================
print("🧠 Bước 4: Khởi tạo các biến Domain Knowledge (Thủ Top 11, Two-Way, eFG%, Superstar)...")

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
# [BỔ SUNG HƯỚNG 4] DBSCAN PHÂN CỤM VƯƠNG TRIỀU (DYNASTY CLUSTERS)
# ==========================================
print("🔮 Bước 4.5: Áp dụng DBSCAN phân cụm mật độ để tìm profile các đội thống trị...")
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Đảm bảo các đặc trưng chiến lược đầu vào cho DBSCAN đều đã được tính toán xong
features_for_clustering = ['SRS', 'NRtg', 'TS%', 'Superstar_Impact_Score']
existing_clust_features = [c for c in features_for_clustering if c in df.columns]

if len(existing_clust_features) == len(features_for_clustering):
    # Trích xuất dữ liệu và chuẩn hóa cục bộ trước khi truyền vào thuật toán mật độ
    X_cluster = df[existing_clust_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X_cluster)

    # Cấu hình eps tối ưu giúp thuật toán bóc tách chính xác các nhóm cực kỳ xuất sắc (Championship Contenders)
    dbscan = DBSCAN(eps=0.6, min_samples=3)
    df['Dynasty_Cluster'] = dbscan.fit_predict(X_scaled)

    # Chuyển đổi thành nhãn nhị phân: 1 nếu lọt vào cụm tinh hoa, 0 nếu bị coi là nhiễu (-1)
    df['Is_Dynasty_Profile'] = (df['Dynasty_Cluster'] != -1).astype(int)
    print(f"   📊 Số lượng đội bóng lọt vào profile Vương Triều: {df['Is_Dynasty_Profile'].sum()} đội.")
else:
    print("⚠️ CẢNH BÁO: Không thể chạy DBSCAN do thiếu một trong các trường đặc trưng cốt lõi.")

# ==========================================
# 6. LƯU DỮ LIỆU SẴN SÀNG CHO MODEL
# ==========================================
print(f"\n💾 Bước 5: Đang xuất tập dữ liệu siêu việt ra file '{OUTPUT_FILE}'...")
if 'Conf_Seed' in df.columns:
    df = df.sort_values(by=['Season_Year', 'Conference', 'Conf_Seed'], ascending=[False, True, True])

df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Đã thêm Xếp hạng (Rank) cho chỉ số Elo Rating, Thể lực & Chiều sâu đội hình.")
print(f"✅ Đã tích hợp thành công bộ lọc phân cụm DBSCAN Dynasty Classifier.")
print(f"✅ Kích thước Dataset cuối cùng: {df.shape[0]} dòng, {df.shape[1]} cột.")