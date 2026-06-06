import os
import time
import warnings
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashteamclutch, teamyearbyyearstats
from nba_api.stats.static import teams

warnings.filterwarnings("ignore")

# 1. Create teams dataset based on the team's data is available.
output_dir = 'data/raw/nba_features_combined'
xls_folder_path = 'data/raw/nba_history'
os.makedirs(output_dir, exist_ok=True)

# List of seasons from 2000/01 to 2025/26
seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2000, 2026)]

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

superstar_history = {
    (2000, 'Los Angeles Lakers'): (1, 2), (2000, 'San Antonio Spurs'): (1, 1), (2000, 'Philadelphia 76ers'): (1, 1), (2000, 'Phoenix Suns'): (1, 1), (2000, 'Miami Heat'): (1, 1), (2000, 'Indiana Pacers'): (0, 1), (2000, 'Utah Jazz'): (0, 1), (2000, 'Charlotte Hornets'): (0, 1), (2000, 'Minnesota Timberwolves'): (0, 1), (2000, 'Sacramento Kings'): (0, 1), (2000, 'Toronto Raptors'): (0, 1),
    (2001, 'Philadelphia 76ers'): (1, 1), (2001, 'San Antonio Spurs'): (1, 1), (2001, 'Los Angeles Lakers'): (1, 2), (2001, 'Sacramento Kings'): (1, 1), (2001, 'Milwaukee Bucks'): (0, 1), (2001, 'Dallas Mavericks'): (0, 1), (2001, 'Phoenix Suns'): (0, 1), (2001, 'Orlando Magic'): (0, 1), (2001, 'Minnesota Timberwolves'): (0, 1), (2001, 'Toronto Raptors'): (0, 1),
    (2002, 'San Antonio Spurs'): (1, 1), (2002, 'New Jersey Nets'): (1, 1), (2002, 'Los Angeles Lakers'): (1, 2), (2002, 'Sacramento Kings'): (1, 1), (2002, 'Orlando Magic'): (1, 1), (2002, 'Dallas Mavericks'): (0, 2), (2002, 'Minnesota Timberwolves'): (0, 1), (2002, 'Seattle SuperSonics'): (0, 1), (2002, 'Boston Celtics'): (0, 1), (2002, 'Detroit Pistons'): (0, 1),
    (2003, 'San Antonio Spurs'): (1, 1), (2003, 'Minnesota Timberwolves'): (1, 1), (2003, 'Los Angeles Lakers'): (1, 2), (2003, 'Orlando Magic'): (1, 1), (2003, 'New Jersey Nets'): (1, 1), (2003, 'Dallas Mavericks'): (0, 2), (2003, 'Sacramento Kings'): (0, 1), (2003, 'Detroit Pistons'): (0, 1), (2003, 'Boston Celtics'): (0, 1), (2003, 'Charlotte Hornets'): (0, 1), (2003, 'Indiana Pacers'): (0, 1), (2003, 'Suns'): (0, 1),
    (2004, 'Minnesota Timberwolves'): (1, 1), (2004, 'San Antonio Spurs'): (1, 1), (2004, 'Indiana Pacers'): (1, 1), (2004, 'Los Angeles Lakers'): (1, 2), (2004, 'Sacramento Kings'): (1, 1), (2004, 'Detroit Pistons'): (0, 1), (2004, 'Dallas Mavericks'): (0, 1), (2004, 'Houston Rockets'): (0, 1), (2004, 'New Jersey Nets'): (0, 1), (2004, 'New Orleans Hornets'): (0, 1), (2004, 'Milwaukee Bucks'): (0, 1),
    (2005, 'Phoenix Suns'): (1, 3), (2005, 'Miami Heat'): (1, 2), (2005, 'Dallas Mavericks'): (1, 1), (2005, 'San Antonio Spurs'): (1, 1), (2005, 'Philadelphia 76ers'): (1, 1), (2005, 'Houston Rockets'): (0, 2), (2005, 'Detroit Pistons'): (0, 1), (2005, 'Cleveland Cavaliers'): (0, 1), (2005, 'Seattle SuperSonics'): (0, 1),
    (2006, 'Phoenix Suns'): (1, 2), (2006, 'Cleveland Cavaliers'): (1, 1), (2006, 'Dallas Mavericks'): (1, 1), (2006, 'Los Angeles Lakers'): (1, 1), (2006, 'Detroit Pistons'): (1, 2), (2006, 'Miami Heat'): (0, 2), (2006, 'San Antonio Spurs'): (0, 1), (2006, 'LA Clippers'): (0, 1), (2006, 'Los Angeles Clippers'): (0, 1),
    (2007, 'Dallas Mavericks'): (1, 1), (2007, 'Phoenix Suns'): (1, 2), (2007, 'Los Angeles Lakers'): (1, 1), (2007, 'San Antonio Spurs'): (1, 1), (2007, 'Cleveland Cavaliers'): (1, 1), (2007, 'Houston Rockets'): (0, 2), (2007, 'Utah Jazz'): (0, 1), (2007, 'Detroit Pistons'): (0, 1), (2007, 'Miami Heat'): (0, 1),
    (2008, 'Los Angeles Lakers'): (1, 1), (2008, 'New Orleans Hornets'): (1, 1), (2008, 'Boston Celtics'): (1, 2), (2008, 'Cleveland Cavaliers'): (1, 1), (2008, 'Orlando Magic'): (1, 1), (2008, 'Utah Jazz'): (0, 1), (2008, 'San Antonio Spurs'): (0, 2), (2008, 'Phoenix Suns'): (0, 2), (2008, 'Dallas Mavericks'): (0, 1),
    (2009, 'Cleveland Cavaliers'): (1, 1), (2009, 'Los Angeles Lakers'): (1, 2), (2009, 'Orlando Magic'): (1, 1), (2009, 'New Orleans Hornets'): (1, 1), (2009, 'Miami Heat'): (1, 1), (2009, 'Houston Rockets'): (0, 1), (2009, 'Boston Celtics'): (0, 1), (2009, 'San Antonio Spurs'): (0, 1), (2009, 'Dallas Mavericks'): (0, 1), (2009, 'Denver Nuggets'): (0, 2),
    (2010, 'Cleveland Cavaliers'): (1, 1), (2010, 'Oklahoma City Thunder'): (1, 1), (2010, 'Los Angeles Lakers'): (1, 2), (2010, 'Orlando Magic'): (1, 1), (2010, 'Dallas Mavericks'): (1, 1), (2010, 'Phoenix Suns'): (0, 2), (2010, 'Utah Jazz'): (0, 1), (2010, 'Miami Heat'): (0, 1), (2010, 'Boston Celtics'): (0, 1), (2010, 'Denver Nuggets'): (0, 1),
    (2011, 'Chicago Bulls'): (1, 1), (2011, 'Orlando Magic'): (1, 1), (2011, 'Miami Heat'): (1, 2), (2011, 'Los Angeles Lakers'): (1, 2), (2011, 'Oklahoma City Thunder'): (1, 2), (2011, 'Dallas Mavericks'): (0, 1), (2011, 'San Antonio Spurs'): (0, 1), (2011, 'New York Knicks'): (0, 2),
    (2012, 'Miami Heat'): (1, 2), (2012, 'Oklahoma City Thunder'): (1, 2), (2012, 'Los Angeles Clippers'): (1, 2), (2012, 'LA Clippers'): (1, 2), (2012, 'Los Angeles Lakers'): (1, 2), (2012, 'San Antonio Spurs'): (1, 1), (2012, 'Minnesota Timberwolves'): (0, 1), (2012, 'New York Knicks'): (0, 2), (2012, 'Boston Celtics'): (0, 1),
    (2013, 'Miami Heat'): (1, 2), (2013, 'Oklahoma City Thunder'): (1, 2), (2013, 'New York Knicks'): (1, 1), (2013, 'Los Angeles Clippers'): (1, 2), (2013, 'LA Clippers'): (1, 2), (2013, 'Los Angeles Lakers'): (1, 2), (2013, 'San Antonio Spurs'): (0, 2), (2013, 'Memphis Grizzlies'): (0, 1), (2013, 'Indiana Pacers'): (0, 1),
    (2014, 'Oklahoma City Thunder'): (1, 1), (2014, 'Miami Heat'): (1, 1), (2014, 'Los Angeles Clippers'): (1, 2), (2014, 'LA Clippers'): (1, 2), (2014, 'Chicago Bulls'): (1, 1), (2014, 'Houston Rockets'): (1, 2), (2014, 'San Antonio Spurs'): (0, 1), (2014, 'Indiana Pacers'): (0, 1), (2014, 'Golden State Warriors'): (0, 1), (2014, 'Portland Trail Blazers'): (0, 2),
    (2015, 'Golden State Warriors'): (1, 2), (2015, 'Houston Rockets'): (1, 1), (2015, 'Cleveland Cavaliers'): (1, 2), (2015, 'Oklahoma City Thunder'): (1, 1), (2015, 'New Orleans Pelicans'): (1, 1), (2015, 'Los Angeles Clippers'): (0, 3), (2015, 'LA Clippers'): (0, 3), (2015, 'Memphis Grizzlies'): (0, 1), (2015, 'San Antonio Spurs'): (0, 1), (2015, 'Portland Trail Blazers'): (0, 1),
    (2016, 'Golden State Warriors'): (1, 3), (2016, 'San Antonio Spurs'): (1, 1), (2016, 'Cleveland Cavaliers'): (1, 1), (2016, 'Oklahoma City Thunder'): (1, 2), (2016, 'Los Angeles Clippers'): (0, 2), (2016, 'LA Clippers'): (0, 2), (2016, 'Toronto Raptors'): (0, 1), (2016, 'Sacramento Kings'): (0, 1), (2016, 'Portland Trail Blazers'): (0, 1),
    (2017, 'Oklahoma City Thunder'): (1, 1), (2017, 'Houston Rockets'): (1, 1), (2017, 'San Antonio Spurs'): (1, 1), (2017, 'Cleveland Cavaliers'): (1, 1), (2017, 'Boston Celtics'): (1, 1), (2017, 'Golden State Warriors'): (0, 2), (2017, 'Utah Jazz'): (0, 1), (2017, 'Milwaukee Bucks'): (0, 1), (2017, 'Washington Wizards'): (0, 1),
    (2018, 'Houston Rockets'): (1, 1), (2018, 'Cleveland Cavaliers'): (1, 1), (2018, 'New Orleans Pelicans'): (1, 1), (2018, 'Portland Trail Blazers'): (1, 1), (2018, 'Oklahoma City Thunder'): (1, 2), (2018, 'Golden State Warriors'): (0, 2), (2018, 'Milwaukee Bucks'): (0, 1), (2018, 'Minnesota Timberwolves'): (0, 2), (2018, 'San Antonio Spurs'): (0, 1), (2018, 'Toronto Raptors'): (0, 1), (2018, 'Indiana Pacers'): (0, 1),
    (2019, 'Milwaukee Bucks'): (1, 1), (2019, 'Houston Rockets'): (1, 1), (2019, 'Oklahoma City Thunder'): (1, 2), (2019, 'Denver Nuggets'): (1, 1), (2019, 'Golden State Warriors'): (1, 2), (2019, 'Toronto Raptors'): (0, 1), (2019, 'Philadelphia 76ers'): (0, 1), (2019, 'Portland Trail Blazers'): (0, 1), (2019, 'Utah Jazz'): (0, 1), (2019, 'Boston Celtics'): (0, 1),
    (2020, 'Milwaukee Bucks'): (1, 1), (2020, 'Los Angeles Lakers'): (1, 2), (2020, 'Houston Rockets'): (1, 2), (2020, 'Dallas Mavericks'): (1, 1), (2020, 'Los Angeles Clippers'): (1, 1), (2020, 'LA Clippers'): (1, 1), (2020, 'Denver Nuggets'): (0, 1), (2020, 'Miami Heat'): (0, 1), (2020, 'Boston Celtics'): (0, 1), (2020, 'Toronto Raptors'): (0, 1), (2020, 'Utah Jazz'): (0, 1), (2020, 'Oklahoma City Thunder'): (0, 1),
    (2021, 'Denver Nuggets'): (1, 1), (2021, 'Philadelphia 76ers'): (1, 1), (2021, 'Golden State Warriors'): (1, 1), (2021, 'Milwaukee Bucks'): (1, 1), (2021, 'Phoenix Suns'): (1, 1), (2021, 'Dallas Mavericks'): (0, 1), (2021, 'Los Angeles Clippers'): (0, 2), (2021, 'LA Clippers'): (0, 2), (2021, 'Utah Jazz'): (0, 1), (2021, 'Portland Trail Blazers'): (0, 1), (2021, 'New York Knicks'): (0, 1), (2021, 'Brooklyn Nets'): (0, 1), (2021, 'Washington Wizards'): (0, 1), (2021, 'Miami Heat'): (0, 1),
    (2022, 'Denver Nuggets'): (1, 1), (2022, 'Philadelphia 76ers'): (1, 1), (2022, 'Milwaukee Bucks'): (1, 1), (2022, 'Phoenix Suns'): (1, 2), (2022, 'Dallas Mavericks'): (1, 1), (2022, 'Boston Celtics'): (0, 1), (2022, 'Golden State Warriors'): (0, 1), (2022, 'Memphis Grizzlies'): (0, 1), (2022, 'Chicago Bulls'): (0, 1), (2022, 'Brooklyn Nets'): (0, 1), (2022, 'Toronto Raptors'): (0, 1), (2022, 'Minnesota Timberwolves'): (0, 1), (2022, 'Atlanta Hawks'): (0, 1),
    (2023, 'Philadelphia 76ers'): (1, 1), (2023, 'Denver Nuggets'): (1, 1), (2023, 'Milwaukee Bucks'): (1, 2), (2023, 'Boston Celtics'): (1, 2), (2023, 'Oklahoma City Thunder'): (1, 1), (2023, 'Dallas Mavericks'): (0, 1), (2023, 'Cleveland Cavaliers'): (0, 1), (2023, 'Golden State Warriors'): (0, 1), (2023, 'Miami Heat'): (0, 1), (2023, 'Sacramento Kings'): (0, 2), (2023, 'New York Knicks'): (0, 1), (2023, 'Los Angeles Lakers'): (0, 1), (2023, 'Portland Trail Blazers'): (0, 1),
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
    'Seattle SuperSonics': 'West', 'New Jersey Nets': 'East', 'New Orleans Hornets': 'West', 'New Orleans/Oklahoma City Hornets': 'West', 'Charlotte Bobcats': 'East', 'Washington Bullets': 'East', 'Vancouver Grizzlies': 'West'
}

# 2. Helper functions
def assign_cascade_targets(row):
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

def assign_superstar_metrics(row):
    year = int(row['Season'])
    team = str(row['Team']).strip()
    has_top5_mvp, all_nba_count = 0, 0
    if (year, team) in superstar_history:
        has_top5_mvp, all_nba_count = superstar_history[(year, team)]
    return pd.Series([has_top5_mvp, all_nba_count])

def standardize_team_name(name):
    if pd.isna(name): return ""
    name = str(name).strip().replace('*', '')
    # Synchronize abbreviations/history to match API data and local files
    mapping = {
        'LA Clippers': 'Los Angeles Clippers',
        'Charlotte Bobcats': 'Charlotte Hornets',
        'New Orleans/Oklahoma City Hornets': 'New Orleans Hornets',
        'Suns': 'Phoenix Suns'
    }
    return mapping.get(name, name)

# Part 1: Downloading hidden stars and play-off history from NBA API
print(f"Collection phase : {seasons[0]} to {seasons[-1]}\n")

print(" Loading the Playoff history of all teams")
nba_teams = teams.get_teams()
playoff_history_dict = {}

for team in nba_teams:
    tid = team['id']
    tname = team['full_name']
    try:
        raw_history = teamyearbyyearstats.TeamYearByYearStats(team_id=tid)
        df_hist = raw_history.get_data_frames()[0]
        for _, row in df_hist.iterrows():
            try:
                y_str = row['YEAR']
                y_int = int(y_str.split('-')[0])
                rounds_won = row.get('PO_ROUND_WON_COUNT', 0)
                if pd.isna(rounds_won): rounds_won = 0
                conf_rank = row.get('CONF_RANK', 15)
                is_champ = row.get('NBA_CHAMPIONSHIP', 'N')
                
                if rounds_won == 4 or is_champ == 'Y': level = 4
                elif rounds_won == 3: level = 4
                elif rounds_won == 2: level = 3
                elif rounds_won == 1: level = 2
                else:
                    level = 1 if conf_rank <= 8 else 0
                playoff_history_dict[(tid, y_int)] = level
            except Exception: pass
    except Exception as e:
        print(f" Can't collect history {tname}: {e}")
    time.sleep(1.0)

print(" Playoff history download complete, begin advanced season-by-season data scan\n")

api_seasons_list = []

for season in seasons:
    print(f"Api season processed: {season}\n")
    year_int = int(season.split('-')[0]) 
    try:
        # REQUEST 1: Momentum (Post All-Star Advanced)
        raw_momentum = leaguedashteamstats.LeagueDashTeamStats(
            season=season, season_type_all_star='Regular Season',
            season_segment_nullable='Post All-Star', measure_type_detailed_defense='Advanced'
        )
        df_momentum = raw_momentum.get_data_frames()[0][['TEAM_ID', 'TEAM_NAME', 'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'TS_PCT']].copy()
        df_momentum.rename(columns={'E_OFF_RATING': 'Post_ORtg', 'E_DEF_RATING': 'Post_DRtg', 'E_NET_RATING': 'Post_NRtg', 'TS_PCT': 'Post_TS_PCT'}, inplace=True)
        time.sleep(2.0)
        
        # REQUEST 2: Per 100 Possessions Base
        raw_team = leaguedashteamstats.LeagueDashTeamStats(
            season=season, season_type_all_star='Regular Season',
            per_mode_detailed='Per100Possessions', measure_type_detailed_defense='Base'
        )
        df_team = raw_team.get_data_frames()[0][['TEAM_ID', 'FG3M', 'AST', 'TOV']].copy()
        df_team.rename(columns={'FG3M': 'Team_3PM_per100', 'AST': 'Team_AST_per100', 'TOV': 'Team_TOV_per100'}, inplace=True)
        time.sleep(2.0)
        
        # REQUEST 3: Opponent Per 100
        raw_opp = leaguedashteamstats.LeagueDashTeamStats(
            season=season, season_type_all_star='Regular Season',
            per_mode_detailed='Per100Possessions', measure_type_detailed_defense='Opponent'
        )
        df_opp = raw_opp.get_data_frames()[0][['TEAM_ID', 'OPP_FG_PCT', 'OPP_FG3_PCT', 'OPP_TOV']].copy()
        df_opp.rename(columns={'OPP_FG_PCT': 'Opp_FG_PCT', 'OPP_FG3_PCT': 'Opp_FG3_PCT', 'OPP_TOV': 'Opp_TOV_forced'}, inplace=True)
        time.sleep(2.0)

        # REQUEST 4: Clutch
        raw_clutch = leaguedashteamclutch.LeagueDashTeamClutch(
            season=season, season_type_all_star='Regular Season',
            per_mode_detailed='PerGame', measure_type_detailed_defense='Base'
        )
        df_clutch = raw_clutch.get_data_frames()[0][['TEAM_ID', 'W_PCT']].copy()
        df_clutch.rename(columns={'W_PCT': 'Clutch_Win_PCT'}, inplace=True)
        time.sleep(2.0)

        # REQUEST 5: Post All-Star Win PCT
        raw_post_base = leaguedashteamstats.LeagueDashTeamStats(
            season=season, season_type_all_star='Regular Season',
            season_segment_nullable='Post All-Star', measure_type_detailed_defense='Base'
        )
        df_post_base = raw_post_base.get_data_frames()[0][['TEAM_ID', 'W_PCT']].copy()
        df_post_base.rename(columns={'W_PCT': 'Post_AllStar_Win_PCT'}, inplace=True)

        # Merge all requests in season
        df_merged = pd.merge(df_momentum, df_team, on='TEAM_ID')
        df_merged = pd.merge(df_merged, df_opp, on='TEAM_ID')
        df_merged = pd.merge(df_merged, df_clutch, on='TEAM_ID')
        df_final = pd.merge(df_merged, df_post_base, on='TEAM_ID')
        
        # Mapping previous season's Playoff experience and assign year
        df_final['Prev_Year_Playoff_Round'] = df_final['TEAM_ID'].map(lambda tid: playoff_history_dict.get((tid, year_int - 1), 0))
        df_final['Season_Year'] = year_int
        
        # Download file from API
        df_final.to_csv(os.path.join(output_dir, f"nba_features_{year_int}.csv"), index=False)
        api_seasons_list.append(df_final)
        print(f" Successfully loaded the season's API metrics.: {season}")
    except Exception as e:
        print(f"Error in the curent season {season}: {e}")
    time.sleep(2.5)

api_master_df = pd.concat(api_seasons_list, ignore_index=True) if api_seasons_list else pd.DataFrame()


# Part II: Read and clean local files (.xls) 
print("Start loading and processing the files .xls")
all_seasons_list = []

if not os.path.exists(xls_folder_path):
    print(f"Error: The folder containing the original data '{xls_folder_path}' is not exist")
else:
    for file_name in os.listdir(xls_folder_path):
        if file_name.endswith('.xls'):
            season_year = int(file_name.split('.')[0])
            file_path = os.path.join(xls_folder_path, file_name)
            
            df_season = pd.read_html(file_path)[0]
            
            if isinstance(df_season.columns, pd.MultiIndex):
                new_cols = []
                for col in df_season.columns:
                    level_0, level_1 = str(col[0]), str(col[1])
                    new_cols.append(level_1 if 'Unnamed' in level_0 else f"{level_0}_{level_1}")
                df_season.columns = new_cols
            
            df_season = df_season.loc[:, ~df_season.columns.str.contains('Unnamed')]
            df_season = df_season[(df_season['Team'] != 'Team') & (df_season['Team'] != 'League Average')].copy()
            
            cols_to_drop = ['Rk', 'Arena', 'Attend.', 'Attend./G']
            df_season = df_season.drop(columns=[col for col in cols_to_drop if col in df_season.columns])
            
            df_season['Is_Playoff'] = df_season['Team'].astype(str).str.contains(r'\*').astype(int)
            df_season['Team'] = df_season['Team'].astype(str).str.replace(r'\*', '', regex=True)
            df_season['Season'] = season_year
            all_seasons_list.append(df_season)

if len(all_seasons_list) > 0:
    print(f" Connection successful {len(all_seasons_list)} season from .xls file.")
    xls_master_df = pd.concat(all_seasons_list, ignore_index=True)
    
    print("Currently working on the engineering aspects (Playoff Labels, Superstars, Conference, Age)")
    xls_master_df[['Target_Top4', 'Target_Top2', 'Target_Champ']] = xls_master_df.apply(assign_cascade_targets, axis=1)
    xls_master_df[['Has_Top5_MVP', 'All_NBA_Count']] = xls_master_df.apply(assign_superstar_metrics, axis=1)
    xls_master_df['Conference'] = xls_master_df['Team'].map(team_conference)
    
    xls_master_df['Age'] = pd.to_numeric(xls_master_df['Age'], errors='coerce')
    champ_mean_age = xls_master_df[xls_master_df['Target_Champ'] == 1]['Age'].mean()
    xls_master_df['Age_Diff'] = abs(xls_master_df['Age'] - champ_mean_age)
    xls_master_df['Age_Rank'] = xls_master_df.groupby('Season')['Age_Diff'].rank(ascending=True, method='min')
    
    if 'Target' in xls_master_df.columns:
        xls_master_df.drop(columns=['Target'], inplace=True)
else:
    print("Never has '.xls' was found to process")
    xls_master_df = pd.DataFrame()


# Part III: Merge all into a single file

if not xls_master_df.empty and not api_master_df.empty:
    
    # Normalize the team name column in both datasets to avoid mismatches during the join process
    xls_master_df['Join_Team'] = xls_master_df['Team'].apply(standardize_team_name)
    api_master_df['Join_Team'] = api_master_df['TEAM_NAME'].apply(standardize_team_name)
    
    # Performing a Left Join preserves all the underlying .xls data structure
    final_dataset = pd.merge(
        xls_master_df,
        api_master_df,
        left_on=['Season', 'Join_Team'],
        right_on=['Season_Year', 'Join_Team'],
        how='left'
    )
    
    # Remove duplicate columns or temporary columns generated during the data merging process
    cols_to_remove = ['Season_Year', 'TEAM_NAME', 'TEAM_ID', 'Join_Team']
    final_dataset.drop(columns=[c for c in cols_to_remove if c in final_dataset.columns], inplace=True)
    
    # Export the final data to a single file named team_dataset.csv in the directory.
    output_filename = 'data/interim/teams_dataset.csv'
    final_dataset.to_csv(output_filename, index=False)
    
    print(f"Total number of rows (Team profile across seasons)): {len(final_dataset)}")
    print(f"The complete merged data file is saved at: '{output_filename}'")
else:
    print("Critical error: One of the two data sources is missing to perform the mergging")