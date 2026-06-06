import pandas as pd
import os

# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN FILE
INPUT_FILE = 'data/processed/final_complete_dataset.csv'
OUTPUT_FILE = 'data/processed/model_ready_dataset.csv'

# 2. ĐỌC DỮ LIỆU
print(f"🔄 Bước 1: Đang đọc dữ liệu từ '{INPUT_FILE}'...")
if not os.path.exists(INPUT_FILE):
    print(f"❌ LỖI: Không tìm thấy file '{INPUT_FILE}'.")
    exit()

df = pd.read_csv(INPUT_FILE)

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
    
    # --- CHỈ SỐ THỂ LỰC & ĐỘI HÌNH (MỚI THÊM) ---
    'Playoff_Core_VORP_Share': False,  # Bộ khung 7 người càng gánh team -> Hạng càng cao
    'Bench_Tactical_BPM': False,       # Ghế dự bị càng xịn -> Hạng càng cao
    'Core_Fatigue_MP': True            # Số phút cày ải THẤP (Ít mệt mỏi nhất) -> Hạng càng cao
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
# 6. LƯU DỮ LIỆU SẴN SÀNG CHO MODEL
# ==========================================
print(f"\n💾 Bước 5: Đang xuất tập dữ liệu siêu việt ra file '{OUTPUT_FILE}'...")
if 'Conf_Seed' in df.columns:
    df = df.sort_values(by=['Season_Year', 'Conference', 'Conf_Seed'], ascending=[False, True, True])

df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Đã thêm Xếp hạng (Rank) cho các chỉ số Thể lực & Chiều sâu đội hình.")
print(f"✅ Kích thước Dataset cuối cùng: {df.shape[0]} dòng, {df.shape[1]} cột.")