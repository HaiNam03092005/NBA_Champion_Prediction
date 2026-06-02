import pandas as pd
import os

# 1. Danh sách lịch sử Playoff
playoff_history = {
    2000: {'champ': 'Los Angeles Lakers', 'runner_up': 'Indiana Pacers', 'cf_losers': ['Portland Trail Blazers', 'New York Knicks']},
    2001: {'champ': 'Los Angeles Lakers', 'runner_up': 'Philadelphia 76ers', 'cf_losers': ['Milwaukee Bucks', 'San Antonio Spurs']},
    2002: {'champ': 'Los Angeles Lakers', 'runner_up': 'New Jersey Nets', 'cf_losers': ['Sacramento Kings', 'Boston Celtics']},
    2003: {'champ': 'San Antonio Spurs', 'runner_up': 'New Jersey Nets', 'cf_losers': ['Dallas Mavericks', 'Detroit Pistons']},
    2004: {'champ': 'Detroit Pistons', 'runner_up': 'Los Angeles Lakers', 'cf_losers': ['Indiana Pacers', 'Minnesota Timberwolves']},
    2005: {'champ': 'San Antonio Spurs', 'runner_up': 'Detroit Pistons', 'cf_losers': ['Miami Heat', 'Phoenix Suns']},
    2006: {'champ': 'Miami Heat', 'runner_up': 'Dallas Mavericks', 'cf_losers': ['Detroit Pistons', 'Phoenix Suns']},
    2007: {'champ': 'San Antonio Spurs', 'runner_up': 'Cleveland Cavaliers', 'cf_losers': ['Detroit Pistons', 'Utah Jazz']},
    2008: {'champ': 'Boston Celtics', 'runner_up': 'Los Angeles Lakers', 'cf_losers': ['Detroit Pistons', 'San Antonio Spurs']},
    2009: {'champ': 'Los Angeles Lakers', 'runner_up': 'Orlando Magic', 'cf_losers': ['Cleveland Cavaliers', 'Denver Nuggets']},
    2010: {'champ': 'Los Angeles Lakers', 'runner_up': 'Boston Celtics', 'cf_losers': ['Orlando Magic', 'Phoenix Suns']},
    2011: {'champ': 'Dallas Mavericks', 'runner_up': 'Miami Heat', 'cf_losers': ['Chicago Bulls', 'Oklahoma City Thunder']},
    2012: {'champ': 'Miami Heat', 'runner_up': 'Oklahoma City Thunder', 'cf_losers': ['Boston Celtics', 'San Antonio Spurs']},
    2013: {'champ': 'Miami Heat', 'runner_up': 'San Antonio Spurs', 'cf_losers': ['Indiana Pacers', 'Memphis Grizzlies']},
    2014: {'champ': 'San Antonio Spurs', 'runner_up': 'Miami Heat', 'cf_losers': ['Indiana Pacers', 'Oklahoma City Thunder']},
    2015: {'champ': 'Golden State Warriors', 'runner_up': 'Cleveland Cavaliers', 'cf_losers': ['Atlanta Hawks', 'Houston Rockets']},
    2016: {'champ': 'Cleveland Cavaliers', 'runner_up': 'Golden State Warriors', 'cf_losers': ['Toronto Raptors', 'Oklahoma City Thunder']},
    2017: {'champ': 'Golden State Warriors', 'runner_up': 'Cleveland Cavaliers', 'cf_losers': ['Boston Celtics', 'San Antonio Spurs']},
    2018: {'champ': 'Golden State Warriors', 'runner_up': 'Cleveland Cavaliers', 'cf_losers': ['Boston Celtics', 'Houston Rockets']},
    2019: {'champ': 'Toronto Raptors', 'runner_up': 'Golden State Warriors', 'cf_losers': ['Milwaukee Bucks', 'Portland Trail Blazers']},
    2020: {'champ': 'Los Angeles Lakers', 'runner_up': 'Miami Heat', 'cf_losers': ['Boston Celtics', 'Denver Nuggets']},
    2021: {'champ': 'Milwaukee Bucks', 'runner_up': 'Phoenix Suns', 'cf_losers': ['Atlanta Hawks', 'LA Clippers', 'Los Angeles Clippers']},
    2022: {'champ': 'Golden State Warriors', 'runner_up': 'Boston Celtics', 'cf_losers': ['Miami Heat', 'Dallas Mavericks']},
    2023: {'champ': 'Denver Nuggets', 'runner_up': 'Miami Heat', 'cf_losers': ['Boston Celtics', 'Los Angeles Lakers']},
    2024: {'champ': 'Boston Celtics', 'runner_up': 'Dallas Mavericks', 'cf_losers': ['Indiana Pacers', 'Minnesota Timberwolves']},
    2025: {'champ': 'Unknown', 'runner_up': 'New York Knicks', 'cf_losers': ['Oklahoma City Thunder', 'Cleveland Cavaliers']}
} 
    # Chưa đá xong nên để là Unknown # Runner_up thì chọn 1 trong 2 đội vào chung kết

# 2. Map dữ liệu Siêu sao: (Năm, Tên Đội): (Has_Top5_MVP, All_NBA_Count)
# Những đội không có trong map này tự động nhận giá trị mặc định là (0, 0)
superstar_history = {
    # Mùa giải 2000
    (2000, 'Los Angeles Lakers'): (1, 2), (2000, 'San Antonio Spurs'): (1, 1), (2000, 'Philadelphia 76ers'): (1, 1), (2000, 'Phoenix Suns'): (1, 1), (2000, 'Miami Heat'): (1, 1), (2000, 'Indiana Pacers'): (0, 1), (2000, 'Utah Jazz'): (0, 1), (2000, 'Charlotte Hornets'): (0, 1), (2000, 'Minnesota Timberwolves'): (0, 1), (2000, 'Sacramento Kings'): (0, 1), (2000, 'Toronto Raptors'): (0, 1),
    # Mùa giải 2001
    (2001, 'Philadelphia 76ers'): (1, 1), (2001, 'San Antonio Spurs'): (1, 1), (2001, 'Los Angeles Lakers'): (1, 2), (2001, 'Sacramento Kings'): (1, 1), (2001, 'Milwaukee Bucks'): (0, 1), (2001, 'Dallas Mavericks'): (0, 1), (2001, 'Phoenix Suns'): (0, 1), (2001, 'Orlando Magic'): (0, 1), (2001, 'Minnesota Timberwolves'): (0, 1), (2001, 'Toronto Raptors'): (0, 1),
    # Mùa giải 2002
    (2002, 'San Antonio Spurs'): (1, 1), (2002, 'New Jersey Nets'): (1, 1), (2002, 'Los Angeles Lakers'): (1, 2), (2002, 'Sacramento Kings'): (1, 1), (2002, 'Orlando Magic'): (1, 1), (2002, 'Dallas Mavericks'): (0, 2), (2002, 'Minnesota Timberwolves'): (0, 1), (2002, 'Seattle SuperSonics'): (0, 1), (2002, 'Boston Celtics'): (0, 1), (2002, 'Detroit Pistons'): (0, 1),
    # Mùa giải 2003
    (2003, 'San Antonio Spurs'): (1, 1), (2003, 'Minnesota Timberwolves'): (1, 1), (2003, 'Los Angeles Lakers'): (1, 2), (2003, 'Orlando Magic'): (1, 1), (2003, 'New Jersey Nets'): (1, 1), (2003, 'Dallas Mavericks'): (0, 2), (2003, 'Sacramento Kings'): (0, 1), (2003, 'Detroit Pistons'): (0, 1), (2003, 'Boston Celtics'): (0, 1), (2003, 'Charlotte Hornets'): (0, 1), (2003, 'Indiana Pacers'): (0, 1), (2003, 'Suns'): (0, 1),
    # Mùa giải 2004
    (2004, 'Minnesota Timberwolves'): (1, 1), (2004, 'San Antonio Spurs'): (1, 1), (2004, 'Indiana Pacers'): (1, 1), (2004, 'Los Angeles Lakers'): (1, 2), (2004, 'Sacramento Kings'): (1, 1), (2004, 'Detroit Pistons'): (0, 1), (2004, 'Dallas Mavericks'): (0, 1), (2004, 'Houston Rockets'): (0, 1), (2004, 'New Jersey Nets'): (0, 1), (2004, 'New Orleans Hornets'): (0, 1), (2004, 'Milwaukee Bucks'): (0, 1),
    # Mùa giải 2005
    (2005, 'Phoenix Suns'): (1, 3), (2005, 'Miami Heat'): (1, 2), (2005, 'Dallas Mavericks'): (1, 1), (2005, 'San Antonio Spurs'): (1, 1), (2005, 'Philadelphia 76ers'): (1, 1), (2005, 'Houston Rockets'): (0, 2), (2005, 'Detroit Pistons'): (0, 1), (2005, 'Cleveland Cavaliers'): (0, 1), (2005, 'Seattle SuperSonics'): (0, 1),
    # Mùa giải 2006
    (2006, 'Phoenix Suns'): (1, 2), (2006, 'Cleveland Cavaliers'): (1, 1), (2006, 'Dallas Mavericks'): (1, 1), (2006, 'Los Angeles Lakers'): (1, 1), (2006, 'Detroit Pistons'): (1, 2), (2006, 'Miami Heat'): (0, 2), (2006, 'San Antonio Spurs'): (0, 1), (2006, 'LA Clippers'): (0, 1), (2006, 'Los Angeles Clippers'): (0, 1),
    # Mùa giải 2007
    (2007, 'Dallas Mavericks'): (1, 1), (2007, 'Phoenix Suns'): (1, 2), (2007, 'Los Angeles Lakers'): (1, 1), (2007, 'San Antonio Spurs'): (1, 1), (2007, 'Cleveland Cavaliers'): (1, 1), (2007, 'Houston Rockets'): (0, 2), (2007, 'Utah Jazz'): (0, 1), (2007, 'Detroit Pistons'): (0, 1), (2007, 'Miami Heat'): (0, 1),
    # Mùa giải 2008
    (2008, 'Los Angeles Lakers'): (1, 1), (2008, 'New Orleans Hornets'): (1, 1), (2008, 'Boston Celtics'): (1, 2), (2008, 'Cleveland Cavaliers'): (1, 1), (2008, 'Orlando Magic'): (1, 1), (2008, 'Utah Jazz'): (0, 1), (2008, 'San Antonio Spurs'): (0, 2), (2008, 'Phoenix Suns'): (0, 2), (2008, 'Dallas Mavericks'): (0, 1),
    # Mùa giải 2009
    (2009, 'Cleveland Cavaliers'): (1, 1), (2009, 'Los Angeles Lakers'): (1, 2), (2009, 'Orlando Magic'): (1, 1), (2009, 'New Orleans Hornets'): (1, 1), (2009, 'Miami Heat'): (1, 1), (2009, 'Houston Rockets'): (0, 1), (2009, 'Boston Celtics'): (0, 1), (2009, 'San Antonio Spurs'): (0, 1), (2009, 'Dallas Mavericks'): (0, 1), (2009, 'Denver Nuggets'): (0, 2),
    # Mùa giải 2010
    (2010, 'Cleveland Cavaliers'): (1, 1), (2010, 'Oklahoma City Thunder'): (1, 1), (2010, 'Los Angeles Lakers'): (1, 2), (2010, 'Orlando Magic'): (1, 1), (2010, 'Dallas Mavericks'): (1, 1), (2010, 'Phoenix Suns'): (0, 2), (2010, 'Utah Jazz'): (0, 1), (2010, 'Miami Heat'): (0, 1), (2010, 'Boston Celtics'): (0, 1), (2010, 'Denver Nuggets'): (0, 1),
    # Mùa giải 2011
    (2011, 'Chicago Bulls'): (1, 1), (2011, 'Orlando Magic'): (1, 1), (2011, 'Miami Heat'): (1, 2), (2011, 'Los Angeles Lakers'): (1, 2), (2011, 'Oklahoma City Thunder'): (1, 2), (2011, 'Dallas Mavericks'): (0, 1), (2011, 'San Antonio Spurs'): (0, 1), (2011, 'New York Knicks'): (0, 2),
    # Mùa giải 2012
    (2012, 'Miami Heat'): (1, 2), (2012, 'Oklahoma City Thunder'): (1, 2), (2012, 'Los Angeles Clippers'): (1, 2), (2012, 'LA Clippers'): (1, 2), (2012, 'Los Angeles Lakers'): (1, 2), (2012, 'San Antonio Spurs'): (1, 1), (2012, 'Minnesota Timberwolves'): (0, 1), (2012, 'New York Knicks'): (0, 2), (2012, 'Boston Celtics'): (0, 1),
    # Mùa giải 2013
    (2013, 'Miami Heat'): (1, 2), (2013, 'Oklahoma City Thunder'): (1, 2), (2013, 'New York Knicks'): (1, 1), (2013, 'Los Angeles Clippers'): (1, 2), (2013, 'LA Clippers'): (1, 2), (2013, 'Los Angeles Lakers'): (1, 2), (2013, 'San Antonio Spurs'): (0, 2), (2013, 'Memphis Grizzlies'): (0, 1), (2013, 'Indiana Pacers'): (0, 1),
    # Mùa giải 2014
    (2014, 'Oklahoma City Thunder'): (1, 1), (2014, 'Miami Heat'): (1, 1), (2014, 'Los Angeles Clippers'): (1, 2), (2014, 'LA Clippers'): (1, 2), (2014, 'Chicago Bulls'): (1, 1), (2014, 'Houston Rockets'): (1, 2), (2014, 'San Antonio Spurs'): (0, 1), (2014, 'Indiana Pacers'): (0, 1), (2014, 'Golden State Warriors'): (0, 1), (2014, 'Portland Trail Blazers'): (0, 2),
    # Mùa giải 2015
    (2015, 'Golden State Warriors'): (1, 2), (2015, 'Houston Rockets'): (1, 1), (2015, 'Cleveland Cavaliers'): (1, 2), (2015, 'Oklahoma City Thunder'): (1, 1), (2015, 'New Orleans Pelicans'): (1, 1), (2015, 'Los Angeles Clippers'): (0, 3), (2015, 'LA Clippers'): (0, 3), (2015, 'Memphis Grizzlies'): (0, 1), (2015, 'San Antonio Spurs'): (0, 1), (2015, 'Portland Trail Blazers'): (0, 1),
    # Mùa giải 2016
    (2016, 'Golden State Warriors'): (1, 3), (2016, 'San Antonio Spurs'): (1, 1), (2016, 'Cleveland Cavaliers'): (1, 1), (2016, 'Oklahoma City Thunder'): (1, 2), (2016, 'Los Angeles Clippers'): (0, 2), (2016, 'LA Clippers'): (0, 2), (2016, 'Toronto Raptors'): (0, 1), (2016, 'Sacramento Kings'): (0, 1), (2016, 'Portland Trail Blazers'): (0, 1),
    # Mùa giải 2017
    (2017, 'Oklahoma City Thunder'): (1, 1), (2017, 'Houston Rockets'): (1, 1), (2017, 'San Antonio Spurs'): (1, 1), (2017, 'Cleveland Cavaliers'): (1, 1), (2017, 'Boston Celtics'): (1, 1), (2017, 'Golden State Warriors'): (0, 2), (2017, 'Utah Jazz'): (0, 1), (2017, 'Milwaukee Bucks'): (0, 1), (2017, 'Washington Wizards'): (0, 1),
    # Mùa giải 2018
    (2018, 'Houston Rockets'): (1, 1), (2018, 'Cleveland Cavaliers'): (1, 1), (2018, 'New Orleans Pelicans'): (1, 1), (2018, 'Portland Trail Blazers'): (1, 1), (2018, 'Oklahoma City Thunder'): (1, 2), (2018, 'Golden State Warriors'): (0, 2), (2018, 'Milwaukee Bucks'): (0, 1), (2018, 'Minnesota Timberwolves'): (0, 2), (2018, 'San Antonio Spurs'): (0, 1), (2018, 'Toronto Raptors'): (0, 1), (2018, 'Indiana Pacers'): (0, 1),
    # Mùa giải 2019
    (2019, 'Milwaukee Bucks'): (1, 1), (2019, 'Houston Rockets'): (1, 1), (2019, 'Oklahoma City Thunder'): (1, 2), (2019, 'Denver Nuggets'): (1, 1), (2019, 'Golden State Warriors'): (1, 2), (2019, 'Toronto Raptors'): (0, 1), (2019, 'Philadelphia 76ers'): (0, 1), (2019, 'Portland Trail Blazers'): (0, 1), (2019, 'Utah Jazz'): (0, 1), (2019, 'Boston Celtics'): (0, 1),
    # Mùa giải 2020
    (2020, 'Milwaukee Bucks'): (1, 1), (2020, 'Los Angeles Lakers'): (1, 2), (2020, 'Houston Rockets'): (1, 2), (2020, 'Dallas Mavericks'): (1, 1), (2020, 'Los Angeles Clippers'): (1, 1), (2020, 'LA Clippers'): (1, 1), (2020, 'Denver Nuggets'): (0, 1), (2020, 'Miami Heat'): (0, 1), (2020, 'Boston Celtics'): (0, 1), (2020, 'Toronto Raptors'): (0, 1), (2020, 'Utah Jazz'): (0, 1), (2020, 'Oklahoma City Thunder'): (0, 1),
    # Mùa giải 2021
    (2021, 'Denver Nuggets'): (1, 1), (2021, 'Philadelphia 76ers'): (1, 1), (2021, 'Golden State Warriors'): (1, 1), (2021, 'Milwaukee Bucks'): (1, 1), (2021, 'Phoenix Suns'): (1, 1), (2021, 'Dallas Mavericks'): (0, 1), (2021, 'Los Angeles Clippers'): (0, 2), (2021, 'LA Clippers'): (0, 2), (2021, 'Utah Jazz'): (0, 1), (2021, 'Portland Trail Blazers'): (0, 1), (2021, 'New York Knicks'): (0, 1), (2021, 'Brooklyn Nets'): (0, 1), (2021, 'Washington Wizards'): (0, 1), (2021, 'Miami Heat'): (0, 1),
    # Mùa giải 2022
    (2022, 'Denver Nuggets'): (1, 1), (2022, 'Philadelphia 76ers'): (1, 1), (2022, 'Milwaukee Bucks'): (1, 1), (2022, 'Phoenix Suns'): (1, 2), (2022, 'Dallas Mavericks'): (1, 1), (2022, 'Boston Celtics'): (0, 1), (2022, 'Golden State Warriors'): (0, 1), (2022, 'Memphis Grizzlies'): (0, 1), (2022, 'Chicago Bulls'): (0, 1), (2022, 'Brooklyn Nets'): (0, 1), (2022, 'Toronto Raptors'): (0, 1), (2022, 'Minnesota Timberwolves'): (0, 1), (2022, 'Atlanta Hawks'): (0, 1),
    # Mùa giải 2023
    (2023, 'Philadelphia 76ers'): (1, 1), (2023, 'Denver Nuggets'): (1, 1), (2023, 'Milwaukee Bucks'): (1, 2), (2023, 'Boston Celtics'): (1, 2), (2023, 'Oklahoma City Thunder'): (1, 1), (2023, 'Dallas Mavericks'): (0, 1), (2023, 'Cleveland Cavaliers'): (0, 1), (2023, 'Golden State Warriors'): (0, 1), (2023, 'Miami Heat'): (0, 1), (2023, 'Sacramento Kings'): (0, 2), (2023, 'New York Knicks'): (0, 1), (2023, 'Los Angeles Lakers'): (0, 1), (2023, 'Portland Trail Blazers'): (0, 1),
    # Mùa giải 2024
    (2024, 'Denver Nuggets'): (1, 1), (2024, 'Oklahoma City Thunder'): (1, 1), (2024, 'Dallas Mavericks'): (1, 1), (2024, 'Milwaukee Bucks'): (1, 2), (2024, 'New York Knicks'): (1, 1), (2024, 'Boston Celtics'): (0, 2), (2024, 'Minnesota Timberwolves'): (0, 2), (2024, 'Phoenix Suns'): (0, 2), (2024, 'Los Angeles Lakers'): (0, 2), (2024, 'Los Angeles Clippers'): (0, 1), (2024, 'LA Clippers'): (0, 1), (2024, 'Indiana Pacers'): (0, 1), (2024, 'Sacramento Kings'): (0, 1),
    (2025, 'Denver Nuggets'): (1, 1), (2025, 'Oklahoma City Thunder'): (1, 1), (2025, 'Dallas Mavericks'): (1, 1), (2025, 'Milwaukee Bucks'): (1, 1), (2025, 'Boston Celtics'): (1, 2), (2025, 'Los Angeles Lakers'): (0, 2), (2025, 'New York Knicks'): (0, 1), (2025, 'Minnesota Timberwolves'): (0, 1), (2025, 'Phoenix Suns'): (0, 1), (2025, 'Golden State Warriors'): (0, 1), (2025, 'Sacramento Kings'): (0, 1), (2025, 'Indiana Pacers'): (0, 1), (2025, 'San Antonio Spurs'): (0, 1)       
}

team_conference = {
    'Boston Celtics': 'East', 'Brooklyn Nets': 'East', 'New York Knicks': 'East', 'Philadelphia 76ers': 'East', 'Toronto Raptors': 'East',
    'Chicago Bulls': 'East', 'Cleveland Cavaliers': 'East', 'Detroit Pistons': 'East', 'Indiana Pacers': 'East', 'Milwaukee Bucks': 'East',
    'Atlanta Hawks': 'East', 'Charlotte Hornets': 'East', 'Miami Heat': 'East', 'Orlando Magic': 'East', 'Washington Wizards': 'East',
    'Denver Nuggets': 'West', 'Minnesota Timberwolves': 'West', 'Oklahoma City Thunder': 'West', 'Portland Trail Blazers': 'West', 'Utah Jazz': 'West',
    'Golden State Warriors': 'West', 'LA Clippers': 'West', 'Los Angeles Clippers': 'West', 'Los Angeles Lakers': 'West', 'Phoenix Suns': 'West', 'Sacramento Kings': 'West',
    'Dallas Mavericks': 'West', 'Houston Rockets': 'West', 'Memphis Grizzlies': 'West', 'New Orleans Pelicans': 'West', 'San Antonio Spurs': 'West',
    # Các đội lịch sử:
    'Seattle SuperSonics': 'West', 'New Jersey Nets': 'East', 'New Orleans Hornets': 'West', 'New Orleans/Oklahoma City Hornets': 'West', 'Charlotte Bobcats': 'East', 'Washington Bullets': 'East', 'Vancouver Grizzlies': 'West'
}

def assign_cascade_targets(row):
    # Dùng đúng tên cột 'Season' và 'Team' của file data gốc
    year = int(row['Season']) 
    team = str(row['Team']).strip()
    
    target_top4, target_top2, target_champ = 0, 0, 0
    
    if year in playoff_history:
        history = playoff_history[year]
        if team == history['champ']:
            target_champ, target_top2, target_top4 = 1, 1, 1
        elif team == history['runner_up']:
            target_top2, target_top4 = 1, 1
        elif team in history['cf_losers']:
            target_top4 = 1
            
    return pd.Series([target_top4, target_top2, target_champ])


# 3. Hàm gán chỉ số Siêu sao mới thêm vào
def assign_superstar_metrics(row):
    year = int(row['Season'])
    team = str(row['Team']).strip()
    
    # Mặc định bằng 0 nếu đội bóng đó không sở hữu siêu sao nào được vinh danh năm đó
    has_top5_mvp, all_nba_count = 0, 0
    
    if (year, team) in superstar_history:
        has_top5_mvp, all_nba_count = superstar_history[(year, team)]
        
    return pd.Series([has_top5_mvp, all_nba_count])


# Đường dẫn thư mục chứa dataset gốc
folder_path = 'data/nba_history'

all_seasons_list = []

# Đọc và làm sạch từng file trong nba_history
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xls'):
        season_year = int(file_name.split('.')[0])
        file_path = os.path.join(folder_path, file_name)
        
        # Đọc dữ liệu bảng HTML
        df_season = pd.read_html(file_path)[0]
        
        # Flatten MultiIndex gộp các cột tầng trên và dưới
        if isinstance(df_season.columns, pd.MultiIndex):
            new_cols = []
            for col in df_season.columns:
                level_0, level_1 = str(col[0]), str(col[1])
                if 'Unnamed' in level_0:
                    new_cols.append(level_1)
                else:
                    new_cols.append(f"{level_0}_{level_1}")
            df_season.columns = new_cols
        
        # Xóa các cột chứa Unnamed sau khi gộp tầng
        df_season = df_season.loc[:, ~df_season.columns.str.contains('Unnamed')]
        
        # Loại bỏ các hàng lặp tiêu đề và hàng trung bình giải đấu
        df_season = df_season[df_season['Team'] != 'Team'].copy()
        df_season = df_season[df_season['Team'] != 'League Average'].copy()
        
        # Bỏ qua các cột không mang tính chiến thuật
        cols_to_drop = ['Rk', 'Arena', 'Attend.', 'Attend./G']
        df_season = df_season.drop(columns=[col for col in cols_to_drop if col in df_season.columns])
        
        # Gắn nhãn đội được vào Playoff (bằng cách kiểm tra ký tự '*')
        df_season['Is_Playoff'] = df_season['Team'].astype(str).str.contains(r'\*').astype(int)
        df_season['Team'] = df_season['Team'].astype(str).str.replace(r'\*', '', regex=True)
        
        # Đính kèm cột năm mùa giải
        df_season['Season'] = season_year
        all_seasons_list.append(df_season)

# Hợp nhất danh sách các DataFrame mùa giải thành một
if len(all_seasons_list) > 0:
    print(f"📦 Đã nối thành công {len(all_seasons_list)} mùa giải.")
    master_df = pd.concat(all_seasons_list, ignore_index=True)
    
    # 1. Gắn nhãn Playoff
    print("🎯 Đang gắn hệ thống 3 nhãn phân cấp (Top 4, Top 2, Champ)...")
    master_df[['Target_Top4', 'Target_Top2', 'Target_Champ']] = master_df.apply(assign_cascade_targets, axis=1)
    
    # 2. Gắn chỉ số Siêu sao
    print("⭐ Đang bổ sung Chỉ số Siêu sao (Has_Top5_MVP, All_NBA_Count)...")
    master_df[['Has_Top5_MVP', 'All_NBA_Count']] = master_df.apply(assign_superstar_metrics, axis=1)
    
    # 3. Gắn nhãn Miền Đông / Miền Tây
    print("🗺️ Đang phân bổ các đội vào Miền Đông (East) và Miền Tây (West)...")
    master_df['Conference'] = master_df['Team'].map(team_conference)
    
    # 4. Tính toán Độ tuổi Hoàng kim (Age_Rank)
    print("🧬 Đang tính toán Độ tuổi Hoàng kim và Xếp hạng Age_Rank...")
    # Chuyển cột Age sang dạng số nguyên/thực để tính toán
    master_df['Age'] = pd.to_numeric(master_df['Age'], errors='coerce')
    
    # Tính tuổi trung bình của các nhà vô địch (Bỏ qua năm 2025 vì chưa có vô địch)
    champ_mean_age = master_df[master_df['Target_Champ'] == 1]['Age'].mean()
    print(f"   => Độ tuổi trung bình của nhà vô địch lịch sử là: {champ_mean_age:.2f} tuổi")
    
    # Tính độ lệch tuổi của từng đội so với tuổi vô địch
    master_df['Age_Diff'] = abs(master_df['Age'] - champ_mean_age)
    
    # Xếp hạng Age_Rank theo từng mùa giải (Đội có độ lệch nhỏ nhất = Hạng 1)
    master_df['Age_Rank'] = master_df.groupby('Season')['Age_Diff'].rank(ascending=True, method='min')
    
    # Dọn dẹp cột thừa
    if 'Target' in master_df.columns:
        master_df.drop(columns=['Target'], inplace=True)
        
    # Xuất file cuối cùng
    master_df.to_csv('data/master_dataset.csv', index=False)
    print("✅ HOÀN TẤT: Dữ liệu đã được lưu tại 'data/master_dataset.csv'.")
else:
    print("❌ Thư mục trống, không tìm thấy file '.xls' nào!")