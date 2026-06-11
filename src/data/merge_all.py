import os
import pandas as pd

# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN CÁC FILE DỮ LIỆU
PATH_MASTER_INPUT    = 'data/interim/teams_dataset.csv'             # File Đội bóng gốc
PATH_PLAYER_INPUT    = 'data/interim/player_data_playoff_ready.csv'  # File Cầu thủ khỏe mạnh/chấn thương
PATH_API_DIFF_INPUT  = 'data/raw/nba_api_diff_features.csv'          # File Chỉ số đối kháng từ NBA API
PATH_FINAL_OUTPUT    = 'data/processed/final_complete_dataset.csv'   # File đích hoàn chỉnh cuối cùng

print("🚀 BẮT ĐẦU PIPELINE GỘP MASTER TỔNG HỢP (TEAM + PLAYERS + NBA API) 🚀")
print("-" * 70)

# ==========================================================
# STEP 1: ĐỌC VÀ PHÂN TÍCH THỂ LỰC & ĐỘ SÂU ĐỘI HÌNH CẦU THỦ
# ==========================================================
print("Bước 1: Đang tải và phân tích dữ liệu đội hình thi đấu (Rotation & Fatigue)...")
if not os.path.exists(PATH_PLAYER_INPUT):
    print(f"❌ LỖI: Không tìm thấy file dữ liệu cầu thủ '{PATH_PLAYER_INPUT}'")
    exit()

df_players = pd.read_csv(PATH_PLAYER_INPUT)
team_season_features = []

for (year, team), group in df_players.groupby(['Season_Year', 'TEAM_NAME']):
    # Sắp xếp cầu thủ theo chỉ số đóng góp VORP giảm dần
    group_sorted = group.sort_values(by='VORP', ascending=False).reset_index(drop=True)
    
    total_team_vorp = group_sorted['VORP'].sum()
    playoff_ready_players = len(group)
    team_avg_obpm = group['OBPM'].mean()
    team_avg_dbpm = group['DBPM'].mean()
    
    # Lấy thông số Top 7 cầu thủ gánh đội (Vòng xoay Playoff chính)
    top7 = group_sorted.head(7)
    top7_vorp = top7['VORP'].sum()
    top7_vorp_share = top7_vorp / total_team_vorp if total_team_vorp != 0 else 0
    
    # Tính toán mức độ quá tải thời gian thi đấu của Top 5 Core chính
    top5_minutes = group_sorted.sort_values(by='Regular_Season_MP', ascending=False).head(5)
    total_core_mp = top5_minutes['Regular_Season_MP'].sum() 
    
    # Đo lường chất lượng đội hình dự bị (Vị trí rotation số 8, 9, 10)
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
print(f"🔹 Đã hoàn tất xử lý độ sâu đội hình cho {df_players_agg.shape[0]} lượt đội bóng.")

# ==========================================================
# STEP 2: ĐỌC VÀ CHUẨN HÓA ĐỒNG BỘ FILE ĐỘI BÓNG GỐC
# ==========================================================
print("\nBước 2: Đang tải và chuẩn hóa cấu trúc file teams_dataset.csv...")
if not os.path.exists(PATH_MASTER_INPUT):
    print(f"❌ LỖI: Không tìm thấy file đội bóng master thô '{PATH_MASTER_INPUT}'")
    exit()

df_master_orig = pd.read_csv(PATH_MASTER_INPUT)

# Chuẩn hóa đồng bộ tên tiêu đề cột
if 'Team' in df_master_orig.columns: df_master_orig.rename(columns={'Team': 'TEAM_NAME'}, inplace=True)
if 'Year' in df_master_orig.columns: df_master_orig.rename(columns={'Year': 'Season_Year'}, inplace=True)
if 'Season' in df_master_orig.columns: df_master_orig.rename(columns={'Season': 'Season_Year'}, inplace=True)

# Từ điển đồng bộ tên viết tắt/lịch sử câu lạc bộ NBA tránh lệch Key khi Merge
team_name_mapping = {
    'Los Angeles Clippers': 'LA Clippers',
    'Charlotte Bobcats': 'Charlotte Hornets',
    'New Jersey Nets': 'Brooklyn Nets',
    'New Orleans/Oklahoma City Hornets': 'New Orleans Hornets'
}
df_master_orig['TEAM_NAME'] = df_master_orig['TEAM_NAME'].replace(team_name_mapping)
df_players_agg['TEAM_NAME'] = df_players_agg['TEAM_NAME'].replace(team_name_mapping)

# ==========================================================
# STEP 3: THỰC HIỆN TRỘN THÔNG TIN ĐỘI BÓNG VÀ CẦU THỦ
# ==========================================================
print("\nBước 3: Đang tiến hành lắp ráp dữ liệu Team Stats và Player Features...")
df_final_complete = pd.merge(df_master_orig, df_players_agg, on=['TEAM_NAME', 'Season_Year'], how='left')

# Điền 0 nếu đội bóng đó bị khuyết dữ liệu phân tích chi tiết cầu thủ lẻ
cols_to_fillna = ['Playoff_Ready_Players', 'Team_Total_VORP', 'Team_Avg_OBPM', 'Team_Avg_DBPM', 
                  'Playoff_Core_VORP_Share', 'Core_Fatigue_MP', 'Bench_Tactical_BPM']
df_final_complete[cols_to_fillna] = df_final_complete[cols_to_fillna].fillna(0)

# ==========================================================
# STEP 4: GỘP THÊM CHỈ SỐ ĐỐI KHÁNG TỪ FILE RAW NBA API
# ==========================================================
print("\nBước 4: Đang tích hợp các chỉ số hiệu số đối kháng từ NBA API...")
if os.path.exists(PATH_API_DIFF_INPUT):
    df_api_diff = pd.read_csv(PATH_API_DIFF_INPUT)
    df_api_diff['TEAM_NAME'] = df_api_diff['TEAM_NAME'].replace(team_name_mapping)
    
    # Chống trùng lặp cột nếu file tổng chạy đi chạy lại nhiều lần
    target_api_cols = ['Diff_PTS', 'Diff_eFG', 'Diff_REB', 'Diff_TOV']
    df_final_complete = df_final_complete.drop(columns=[c for c in target_api_cols if c in df_final_complete.columns], errors='ignore')
    
    # Tiến hành gộp
    df_final_complete = pd.merge(df_final_complete, df_api_diff, on=['Season_Year', 'TEAM_NAME'], how='left')
    df_final_complete[target_api_cols] = df_final_complete[target_api_cols].fillna(0)
    print("🔹 Đã gộp thành công 4 đặc trưng đối kháng: Diff_PTS, Diff_eFG, Diff_REB, Diff_TOV.")
else:
    print("⚠️ CẢNH BÁO: Chưa tìm thấy file 'nba_api_diff_features.csv'. Bỏ qua tích hợp chỉ số API.")

# ==========================================================
# STEP 5: XUẤT FILE DATASET HOÀN CHỈNH ĐỂ HUẤN LUYỆN
# ==========================================================
print(f"\nBước 5: Đang đóng gói và lưu bộ dữ liệu tổng hợp vào '{PATH_FINAL_OUTPUT}'...")
os.makedirs(os.path.dirname(PATH_FINAL_OUTPUT), exist_ok=True)
df_final_complete.to_csv(PATH_FINAL_OUTPUT, index=False)

print("-" * 70)
print("✅ HOÀN THÀNH MỸ MÃN TIẾN TRÌNH GỘP TỔNG!")
print(f"📊 Bộ dữ liệu hoàn chỉnh đạt kích thước: {df_final_complete.shape[0]} hàng, {df_final_complete.shape[1]} cột.")