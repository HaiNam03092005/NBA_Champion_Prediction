import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashteamclutch, teamyearbyyearstats
from nba_api.stats.static import teams
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# 1. TẠO THƯ MỤC CHỨA DỮ LIỆU TỔNG HỢP
# ==========================================================
output_dir = 'data/nba_features_combined'
os.makedirs(output_dir, exist_ok=True)

# Tạo danh sách các mùa giải từ 2000-01 đến 2024-25
seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2000, 2026)]

print("======================================================================")
print("🏀 KHỞI ĐỘNG HỆ THỐNG CÀO DỮ LIỆU NÂNG CAO (HỆ THỐNG 5 REQUESTS) 🏀")
print("======================================================================")
print(f"📅 Giai đoạn thu thập: {seasons[0]} đến {seasons[-1]}")
print(f"📂 Thư mục lưu trữ: {output_dir}\n")


# ==========================================================
# 2. BƯỚC CHUẨN BỊ: TỰ ĐỘNG CÀO LỊCH SỬ PLAYOFF CỦA 30 ĐỘI BÓNG
# ==========================================================
print("⏳ BƯỚC 1: Đang tải lịch sử Playoff toàn bộ các đội (Pre-fetching)...")
nba_teams = teams.get_teams()
playoff_history_dict = {} # Key: (team_id, year), Value: Thang điểm 0-4

for team in nba_teams:
    tid = team['id']
    tname = team['full_name']
    print(f"   📥 Đang lấy lịch sử Playoff: {tname}...")
    
    try:
        # Lấy toàn bộ lịch sử từ trước đến nay của đội bóng trong 1 request duy nhất
        raw_history = teamyearbyyearstats.TeamYearByYearStats(team_id=tid)
        df_hist = raw_history.get_data_frames()[0]
        
        for _, row in df_hist.iterrows():
            try:
                y_str = row['YEAR'] # Định dạng '2000-01'
                y_int = int(y_str.split('-')[0]) # Chuyển thành số năm 2000
                
                rounds_won = row.get('PO_ROUND_WON_COUNT', 0)
                if pd.isna(rounds_won): rounds_won = 0
                
                conf_rank = row.get('CONF_RANK', 15)
                is_champ = row.get('NBA_CHAMPIONSHIP', 'N')
                
                # Logic phân cấp thang điểm theo yêu cầu của bạn (0 đến 4)
                if rounds_won == 4 or is_champ == 'Y':
                    level = 4  # Vô địch
                elif rounds_won == 3:
                    level = 4  # Vào tới Finals (Á quân)
                elif rounds_won == 2:
                    level = 3  # Vào tới Chung kết miền (Conf Finals)
                elif rounds_won == 1:
                    level = 2  # Vào tới Bán kết miền (Conf Semis)
                else: # rounds_won == 0
                    # Dựa vào hạng miền Regular Season để biết có được đá Vòng 1 Playoff không
                    if conf_rank <= 8:
                        level = 1  # Bị loại ở Vòng 1 Playoff
                    else:
                        level = 0  # Hụt Playoff (Không lọt vào top 8)
                        
                playoff_history_dict[(tid, y_int)] = level
            except Exception:
                pass
    except Exception as e:
        print(f"   ⚠️ Không thể lấy lịch sử của {tname}: {e}")
        
    time.sleep(1.5) # Tránh bị chặn IP khi quét danh sách đội

print("✅ Hoàn thành tải lịch sử Playoff! Bắt đầu quét dữ liệu theo mùa giải.\n")


# ==========================================================
# 3. VÒNG LẶP QUÉT VÀ TRÍCH XUẤT DỮ LIỆU TỪNG MÙA GIẢI
# ==========================================================
for season in seasons:
    print(f"⚡ ĐANG XỬ LÝ MÙA GIẢI {season} " + "="*35)
    year_int = int(season.split('-')[0]) 
    
    try:
        # --- REQUEST 1: DỮ LIỆU PHONG ĐỘ NÂNG CAO (POST ALL-STAR ADVANCED) ---
        print(" 📊 [1/5] Đang tải chỉ số ẩn Momentum (Post All-Star Advanced)...")
        raw_momentum = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            season_segment_nullable='Post All-Star',
            measure_type_detailed_defense='Advanced',
        )
        df_momentum = raw_momentum.get_data_frames()[0][['TEAM_ID', 'TEAM_NAME', 'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'TS_PCT']].copy()
        df_momentum.rename(columns={
            'E_OFF_RATING': 'Post_ORtg',
            'E_DEF_RATING': 'Post_DRtg',
            'E_NET_RATING': 'Post_NRtg',
            'TS_PCT': 'Post_TS_PCT'
        }, inplace=True)
        
        time.sleep(2.5)
        
        # --- REQUEST 2: DỮ LIỆU TẤN CÔNG TRUYỀN THỐNG (TEAM PER 100 POSSESSIONS) ---
        print(" 🏀 [2/5] Đang tải dữ liệu Tấn công Per 100 của Đội nhà...")
        raw_team = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='Per100Possessions',
            measure_type_detailed_defense='Base',
        )
        df_team = raw_team.get_data_frames()[0][['TEAM_ID', 'FG3M', 'AST', 'TOV']].copy()
        df_team.rename(columns={
            'FG3M': 'Team_3PM_per100',
            'AST': 'Team_AST_per100',
            'TOV': 'Team_TOV_per100'
        }, inplace=True)
        
        time.sleep(2.5)
        
        # --- REQUEST 3: DỮ LIỆU PHÒNG THỦ (OPPONENT PER 100 POSSESSIONS) ---
        print(" 🛡️ [3/5] Đang tải dữ liệu Phòng thủ (Chỉ số của Đối thủ)...")
        raw_opp = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='Per100Possessions',
            measure_type_detailed_defense='Opponent',
        )
        df_opp = raw_opp.get_data_frames()[0][['TEAM_ID', 'OPP_FG_PCT', 'OPP_FG3_PCT', 'OPP_TOV']].copy()
        df_opp.rename(columns={
            'OPP_FG_PCT': 'Opp_FG_PCT',
            'OPP_FG3_PCT': 'Opp_FG3_PCT',
            'OPP_TOV': 'Opp_TOV_forced'
        }, inplace=True)

        time.sleep(2.5)

        # --- REQUEST 4: DỮ LIỆU BẢN LĨNH PHÚT CUỐI (CLUTCH WIN PERCENTAGE) ---
        print(" 🎯 [4/5] Đang tải dữ liệu Bản lĩnh Clutch phút cuối...")
        raw_clutch = leaguedashteamclutch.LeagueDashTeamClutch(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base'
        )
        df_clutch = raw_clutch.get_data_frames()[0][['TEAM_ID', 'W_PCT']].copy()
        df_clutch.rename(columns={'W_PCT': 'Clutch_Win_PCT'}, inplace=True)

        time.sleep(2.5)

        # --- REQUEST 5: TỶ LỆ THẮNG GIAI ĐOẠN NƯỚC RÚT (POST ALL-STAR WIN PCT) ---
        print(" 📈 [5/5] Đang tải Tỷ lệ thắng chặng nước rút (Post All-Star Win PCT)...")
        raw_post_base = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            season_segment_nullable='Post All-Star',
            measure_type_detailed_defense='Base',
        )
        df_post_base = raw_post_base.get_data_frames()[0][['TEAM_ID', 'W_PCT']].copy()
        df_post_base.rename(columns={'W_PCT': 'Post_AllStar_Win_PCT'}, inplace=True)


        # --- HỢP NHẤT TẤT CẢ CÁC REQUESTS DỮ LIỆU NĂM NÀY ---
        df_merged = pd.merge(df_momentum, df_team, on='TEAM_ID')
        df_merged = pd.merge(df_merged, df_opp, on='TEAM_ID')
        df_merged = pd.merge(df_merged, df_clutch, on='TEAM_ID')
        df_final = pd.merge(df_merged, df_post_base, on='TEAM_ID')
        
        # --- ÁNH XẠ (MAP) DỮ LIỆU KINH NGHIỆM PLAYOFF MÙA TRƯỚC ---
        # Logic: Lấy dữ liệu Playoff của năm (year_int - 1) để gán vào năm hiện tại year_int
        df_final['Prev_Year_Playoff_Round'] = df_final['TEAM_ID'].map(
            lambda tid: playoff_history_dict.get((tid, year_int - 1), 0)
        )
        
        # Gắn nhãn năm chuẩn cho mùa giải
        df_final['Season_Year'] = year_int
        
        # Xuất file CSV
        file_name = f"nba_features_{year_int}.csv"
        df_final.to_csv(os.path.join(output_dir, file_name), index=False)
        print(f"💾 THÀNH CÔNG: Đã xuất tập tính năng mở rộng đầy đủ -> {file_name}")
        
    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG tại mùa giải {season}: {e}")
    
    time.sleep(3.0) # Độ trễ an toàn giữa các mùa giải nhằm bảo vệ VPN
    print()

print("="*70)
print("🎯 TIẾN TRÌNH HOÀN TẤT TRỌN VẸN! ĐÃ BỔ SUNG ĐỦ CLUTCH, MOMENTUM & PLAYOFF DNA")
print("="*70)