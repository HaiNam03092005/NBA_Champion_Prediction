import pandas as pd
import time
import os
from nba_api.stats.endpoints import leaguegamelog

def calculate_efg(fgm, fg3m, fga):
    """Tính Effective Field Goal Percentage (eFG%)"""
    if fga == 0: return 0
    return (fgm + 0.5 * fg3m) / fga

# Cấu hình năm chạy trực tiếp
start_year = 2000
end_year = 2025

print("🚀 BẮT ĐẦU PIPELINE: TẢI DỮ LIỆU NBA API & TRÍCH XUẤT ĐẶC TRƯNG 🚀")
print("-" * 70)

# Đảm bảo thư mục lưu trữ thô tồn tại
os.makedirs('data/raw', exist_ok=True)
all_games_list = []

# ==========================================================
# 1. TẢI DỮ LIỆU GAME-BY-GAME TỪ API
# ==========================================================
for year in range(start_year, end_year + 1):
    season_str = f"{year}-{str(year+1)[-2:]}"
    print(f"📥 Đang tải dữ liệu Regular Season {season_str}...")
    
    try:
        game_log = leaguegamelog.LeagueGameLog(
            season=season_str, 
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        
        game_log['Season_Year'] = year + 1 
        all_games_list.append(game_log)
        
        # Nghỉ 1.5 giây giữa mỗi mùa giải để tránh bị hệ thống NBA chặn IP
        time.sleep(1.5) 
    except Exception as e:
        print(f"⚠️ Lỗi hoặc không có dữ liệu mùa {season_str}: {e}")
        
if not all_games_list:
    print("❌ Không tải được dữ liệu nào từ NBA API. Dừng tiến trình!")
    exit()
    
# Gộp log tất cả các mùa giải đã tải
raw_games = pd.concat(all_games_list, ignore_index=True)
raw_games['eFG_pct'] = raw_games.apply(lambda row: calculate_efg(row['FGM'], row['FG3M'], row['FGA']), axis=1)

# ==========================================================
# 2. TÍNH TOÁN CÁC CHỈ SỐ CHÊNH LỆCH (DIFFERENTIAL STATS) & H2H
# ==========================================================
print("🔄 Đang phân tích đối kháng để tính toán Differential Stats...")
cols_to_keep = ['GAME_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'Season_Year', 'PTS', 'eFG_pct', 'REB', 'TOV']
df_main = raw_games[cols_to_keep].copy()

# Ghép cặp song song để tìm thông số của đối thủ trong cùng 1 trận
df_merged = pd.merge(df_main, df_main, on=['GAME_ID', 'Season_Year'], suffixes=('_Team', '_Opp'))
df_games_h2h = df_merged[df_merged['TEAM_ABBREVIATION_Team'] != df_merged['TEAM_ABBREVIATION_Opp']].copy()

# Tính trung bình mùa giải
team_season_stats = df_games_h2h.groupby(['Season_Year', 'TEAM_NAME_Team']).agg(
    PTS_For=('PTS_Team', 'mean'), PTS_Against=('PTS_Opp', 'mean'),
    eFG_For=('eFG_pct_Team', 'mean'), eFG_Against=('eFG_pct_Opp', 'mean'),
    REB_For=('REB_Team', 'mean'), REB_Against=('REB_Opp', 'mean'),
    TOV_For=('TOV_Team', 'mean'), TOV_Against=('TOV_Opp', 'mean')
).reset_index()

# Lập công thức tính hiệu số (Differential)
team_season_stats['Diff_PTS'] = team_season_stats['PTS_For'] - team_season_stats['PTS_Against']
team_season_stats['Diff_eFG'] = team_season_stats['eFG_For'] - team_season_stats['eFG_Against']
team_season_stats['Diff_REB'] = team_season_stats['REB_For'] - team_season_stats['REB_Against']
team_season_stats['Diff_TOV'] = team_season_stats['TOV_For'] - team_season_stats['TOV_Against']

team_season_stats.rename(columns={'TEAM_NAME_Team': 'TEAM_NAME'}, inplace=True)
diff_df = team_season_stats[['Season_Year', 'TEAM_NAME', 'Diff_PTS', 'Diff_eFG', 'Diff_REB', 'Diff_TOV']]

print("📊 Đang tính toán ma trận đối đầu trực tiếp (H2H Matrix)...")
df_games_h2h['Win'] = (df_games_h2h['PTS_Team'] > df_games_h2h['PTS_Opp']).astype(int)
df_games_h2h['Margin'] = df_games_h2h['PTS_Team'] - df_games_h2h['PTS_Opp']

h2h_matrix = df_games_h2h.groupby(['Season_Year', 'TEAM_NAME_Team', 'TEAM_NAME_Opp']).agg(
    H2H_Win_Rate=('Win', 'mean'),
    H2H_Point_Margin=('Margin', 'mean')
).reset_index()

h2h_matrix.rename(columns={'TEAM_NAME_Team': 'TEAM_NAME', 'TEAM_NAME_Opp': 'OPPONENT_NAME'}, inplace=True)

# Xuất các file đặc trưng ra thư mục thô để merge_all nạp vào sau
h2h_matrix.to_csv('data/raw/nba_api_h2h_matrix.csv', index=False)
print("💾 Đã tạo ma trận đối đầu tại 'data/raw/nba_api_h2h_matrix.csv'")

diff_df.to_csv('data/raw/nba_api_diff_features.csv', index=False)
print("💾 Đã tạo đặc trưng đối kháng tại 'data/raw/nba_api_diff_features.csv'")

print("-" * 70)
print("✅ THÀNH CÔNG! Toàn bộ dữ liệu đặc trưng NBA API đã được trích xuất.")