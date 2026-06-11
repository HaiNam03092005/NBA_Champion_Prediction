import pandas as pd
import numpy as np
import os
import glob
import re
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
STATS_RAW_FILE = 'data/raw/player_data_raw.csv'
INJURY_FOLDER = 'data/raw/injury_history'
INJURY_REPORTS_FILE = 'data/raw/injury_reports.csv'

OUTPUT_PLAYER_FILE = 'data/interim/player_data_playoff_ready.csv'
OUTPUT_TEAM_INJURY_FILE = 'data/interim/team_injury_factors.csv'

# ==========================================
# CÁC HÀM HỖ TRỢ DỮ LIỆU
# ==========================================
def normalize_name(name):
    if pd.isna(name): return ""
    n = str(name).lower()
    n = re.sub(r'[^a-z\s]', '', n) 
    n = n.replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '')
    return re.sub(r'\s+', ' ', n).strip()

def extract_player_names(text):
    if pd.isna(text) or not str(text).strip(): return []
    text = re.sub(r'\(.*?\)', '', str(text))  # Xóa phần trong ngoặc
    return [p.strip() for p in re.split(r'[•/]', text) if p.strip()]

def get_season_dates(year):
    if year == 2020: return "2019-10-22", "2020-08-17"
    elif year == 2021: return "2020-12-22", "2021-05-22"
    elif year == 2012: return "2011-12-25", "2012-04-28"
    else: return f"{int(year) - 1}-10-15", f"{int(year)}-04-15"

# ==========================================
# HÀM CHÍNH: XỬ LÝ CẦU THỦ & MÔ HÌNH CHẤN THƯƠNG
# ==========================================
def process_player_data():
    # Xóa màn hình terminal khi bắt đầu chạy
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🚀 BẮT ĐẦU XỬ LÝ CẦU THỦ & MÔ HÌNH CHẤN THƯƠNG XÁC SUẤT 🚀")
    print("-" * 70)

    # 1. ĐỌC VÀ LÀM SẠCH DỮ LIỆU CẦU THỦ GỐC
    if not os.path.exists(STATS_RAW_FILE):
        print(f"❌ LỖI: Không tìm thấy file gốc '{STATS_RAW_FILE}'")
        return None, None

    print(f"📥 Đang tải dữ liệu cầu thủ từ: '{STATS_RAW_FILE}'")
    df_stats = pd.read_csv(STATS_RAW_FILE)

    # Xử lý cầu thủ chuyển nhượng (Lấy team cuối cùng)
    df_players = df_stats.drop_duplicates(subset=['Player', 'Season_Year'], keep='last').copy()
    df_players['Regular_Season_MP'] = pd.to_numeric(df_players.get('Regular_Season_MP', 0), errors='coerce').fillna(0)
    df_players = df_players[df_players['Regular_Season_MP'] >= 250].copy()
    df_players['Norm_Name'] = df_players['Player'].apply(normalize_name)

    # Khởi tạo xác suất ra sân mặc định là 1.0 (Khỏe mạnh) cho tất cả cầu thủ
    df_players['Prob_Play'] = 1.0

    # 2. XỬ LÝ DỮ LIỆU LỊCH SỬ CHẤN THƯƠNG (Quá khứ - Hard Filter)
    if os.path.exists(INJURY_FOLDER):
        print(f"🔎 Đang phân tích lịch sử chấn thương từ thư mục '{INJURY_FOLDER}'...")
        csv_files = glob.glob(os.path.join(INJURY_FOLDER, "*.csv"))
        
        transaction_lists, static_status_list = [], []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                cols_lower = [str(c).lower().strip() for c in df.columns]
                
                if 'acquired' in cols_lower and 'relinquished' in cols_lower:
                    df.columns = df.columns.str.capitalize()
                    valid_df = df[['Date', 'Acquired', 'Relinquished']].copy()
                    valid_df['Date'] = pd.to_datetime(valid_df['Date'], errors='coerce')
                    transaction_lists.append(valid_df.dropna(subset=['Date']))
                elif 'player' in cols_lower and 'update' in cols_lower:
                    df_mapped = pd.DataFrame()
                    p_col = df.columns[cols_lower.index('player')]
                    u_col = df.columns[cols_lower.index('update')]
                    df_mapped['Player'] = df[p_col].astype(str)
                    df_mapped['Season_Year'] = df[u_col].apply(lambda x: int(re.search(r'\d{4}', str(x)).group()) if re.search(r'\d{4}', str(x)) else None)
                    static_status_list.append(df_mapped.dropna(subset=['Season_Year']))
            except Exception as e:
                pass

        df_tx = pd.concat(transaction_lists, ignore_index=True) if transaction_lists else pd.DataFrame()
        df_status = pd.concat(static_status_list, ignore_index=True) if static_status_list else pd.DataFrame()
        if not df_status.empty: df_status['Norm_Name'] = df_status['Player'].apply(normalize_name)

        # Quét xác suất chấn thương trong quá khứ (Gán Prob_Play = 0.0)
        unique_years = sorted(df_players['Season_Year'].unique().astype(int))
        for year in unique_years:
            injured_set = set()
            if not df_tx.empty:
                start_date, playoff_start = get_season_dates(year)
                season_tx = df_tx[(df_tx['Date'] >= start_date) & (df_tx['Date'] <= playoff_start)]
                for _, row in season_tx.iterrows():
                    for p in extract_player_names(row['Relinquished']): injured_set.add(normalize_name(p))
                    for p in extract_player_names(row['Acquired']):
                        norm_p = normalize_name(p)
                        if norm_p in injured_set: injured_set.remove(norm_p)
            
            if not df_status.empty:
                for p in df_status[df_status['Season_Year'] == year]['Norm_Name']: injured_set.add(p)
            
            # Gán xác suất 0.0 cho các cầu thủ chấn thương trong quá khứ
            mask_injured = (df_players['Season_Year'] == year) & (df_players['Norm_Name'].isin(injured_set))
            df_players.loc[mask_injured, 'Prob_Play'] = 0.0

    # 3. CẬP NHẬT TỪ BÁO CÁO CHẤN THƯƠNG HIỆN TẠI (Soft Filter - Xác suất)
    if os.path.exists(INJURY_REPORTS_FILE):
        print(f"🩹 Đang cập nhật báo cáo chấn thương hiện tại từ '{INJURY_REPORTS_FILE}'...")
        injuries_current = pd.read_csv(INJURY_REPORTS_FILE)
        status_prob = {'Available': 1.0, 'Probable': 0.75, 'Questionable': 0.50, 'Doubtful': 0.25, 'Out': 0.0}
        
        # Merge để lấy xác suất cập nhật
        injuries_current['Prob_Play_New'] = injuries_current['Status'].map(status_prob).fillna(1.0)
        injuries_current['Norm_Name'] = injuries_current['Player_Name'].apply(normalize_name)
        
        df_players = pd.merge(df_players, injuries_current[['Season_Year', 'Norm_Name', 'Prob_Play_New']], 
                              on=['Season_Year', 'Norm_Name'], how='left')
        
        # Nếu có xác suất mới từ báo cáo, ưu tiên dùng xác suất mới
        df_players['Prob_Play'] = np.where(df_players['Prob_Play_New'].notna(), df_players['Prob_Play_New'], df_players['Prob_Play'])
        df_players.drop(columns=['Prob_Play_New'], inplace=True)

    # 4. TÍNH TOÁN SỨC MẠNH KỲ VỌNG (EXPECTED METRICS)
    print("🧮 Đang tính toán các chỉ số Sức mạnh Kỳ vọng (Expected Metrics)...")
    if 'VORP' in df_players.columns:
        df_players['Expected_VORP'] = df_players['VORP'] * df_players['Prob_Play']
    if 'BPM' in df_players.columns:
        df_players['Expected_BPM'] = df_players['BPM'] * df_players['Prob_Play']

    # 5. TỔNG HỢP CHỈ SỐ LÊN CẤP ĐỘ ĐỘI BÓNG (TEAM LEVEL)
    team_col = 'Tm' if 'Tm' in df_players.columns else ('TEAM_NAME' if 'TEAM_NAME' in df_players.columns else None)
    
    if team_col:
        team_injury_adjusted = df_players.groupby(['Season_Year', team_col]).agg(
            Team_Health_Factor=('Prob_Play', 'mean'),
            Adjusted_Team_VORP=('Expected_VORP', 'sum') if 'Expected_VORP' in df_players.columns else ('Prob_Play', 'count')
        ).reset_index()
        team_injury_adjusted.rename(columns={team_col: 'TEAM_NAME'}, inplace=True)
        
        os.makedirs('data/interim', exist_ok=True)
        team_injury_adjusted.to_csv(OUTPUT_TEAM_INJURY_FILE, index=False)
        print(f"📊 Đã xuất chỉ số chấn thương/thể lực Đội bóng ra '{OUTPUT_TEAM_INJURY_FILE}'")

    # 6. XUẤT FILE CẦU THỦ CUỐI CÙNG
    df_players.drop(columns=['Norm_Name'], inplace=True, errors='ignore')
    os.makedirs('data/interim', exist_ok=True)
    df_players.to_csv(OUTPUT_PLAYER_FILE, index=False)
    
    print("-" * 70)
    print(f"✅ XONG! Dữ liệu cầu thủ đã sẵn sàng với mô hình chấn thương xác suất.")
    print(f"   Tổng số cầu thủ hợp lệ (kể cả chấn thương): {len(df_players)}")
    
    return df_players

# Gọi trực tiếp hàm xử lý để file tự chạy khi gõ lệnh terminal
process_player_data()