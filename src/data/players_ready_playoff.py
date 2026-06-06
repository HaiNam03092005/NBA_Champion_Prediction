import pandas as pd
import os
import glob
import re

# Declaring the configuration path
stats_raw_file = 'data/raw/player_data_raw.csv'
injury_folder = 'data/raw/injury_history'

os.system('cls' if os.name == 'nt' else 'clear')

# Pre-defined support functions are provided for use below
def normalize_name(name):
    if pd.isna(name): return ""
    n = str(name).lower()
    n = re.sub(r'[^a-z\s]', '', n) 
    n = n.replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '')
    return re.sub(r'\s+', ' ', n).strip()

def extract_player_names(text):
    if pd.isna(text) or not str(text).strip(): return []
    text = re.sub(r'\(.*?\)', '', str(text))  # Delete the part in parentheses (...)
    return [p.strip() for p in re.split(r'[•/]', text) if p.strip()]

def get_season_dates(year):
    if year == 2020: return "2019-10-22", "2020-08-17"
    elif year == 2021: return "2020-12-22", "2021-05-22"
    elif year == 2012: return "2011-12-25", "2012-04-28"
    else: return f"{int(year) - 1}-10-15", f"{int(year)}-04-15"

# Part I: Read and clean the original stats files
if not os.path.exists(stats_raw_file):
    print(f"Error: Original stats file not found '{stats_raw_file}'in the current directory")
    exit()

print(f"Loading data from file: '{stats_raw_file}'")
df_stats = pd.read_csv(stats_raw_file)

# Handling traded players (keeping the player's last stats for that season = second team)
df_stats_clean = df_stats.drop_duplicates(subset=['Player', 'Season_Year'], keep='last').copy()

# Filter by minimum playing time >= 250 minutes
df_stats_clean['Regular_Season_MP'] = pd.to_numeric(df_stats_clean['Regular_Season_MP'], errors='coerce').fillna(0)
df_stats_clean = df_stats_clean[df_stats_clean['Regular_Season_MP'] >= 250].copy()

# Standardize the name column for later reference
df_stats_clean['Norm_Name'] = df_stats_clean['Player'].apply(normalize_name)


# Part II: Scan folders and classify the structure of 3 injury files
if not os.path.exists(injury_folder):
    print(f"Error: Folder not found '{injury_folder}'")
    exit()

csv_files = glob.glob(os.path.join(injury_folder, "*.csv"))
if not csv_files:
    print(f"Folder '{injury_folder}' is empty. There are no injury files to filter")
    exit()

print(f"Found {len(csv_files)} file CSV in folder '{injury_folder}'. Classifying")

transaction_lists = []
static_status_list = []

for file in csv_files:
    filename = os.path.basename(file)
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        cols_lower = [c.lower() for c in df.columns]
        
        # Group 1:Log file (injury_data.csv & NBA Player Injury Stats)
        if 'acquired' in cols_lower and 'relinquished' in cols_lower:
            df.columns = df.columns.str.capitalize()
            valid_df = df[['Date', 'Acquired', 'Relinquished']].copy()
            valid_df['Date'] = pd.to_datetime(valid_df['Date'], errors='coerce')
            transaction_lists.append(valid_df.dropna(subset=['Date']))
            print(f"{filename}: Transaction Log")
            
        # Group 2: Status Report File (sportsref_download.csv)
        elif 'player' in cols_lower and 'update' in cols_lower:
            df_mapped = pd.DataFrame()
            p_col = df.columns[cols_lower.index('player')]
            u_col = df.columns[cols_lower.index('update')]
            
            df_mapped['Player'] = df[p_col].astype(str)
            
            # Extract the year from the date string (Ex: "Mon, Mar 9, 2026" -> 2026)
            def get_year(date_str):
                match = re.search(r'\d{4}', str(date_str))
                return int(match.group()) if match else None
                
            df_mapped['Season_Year'] = df[u_col].apply(get_year)
            static_status_list.append(df_mapped.dropna(subset=['Season_Year']))
            print(f"{filename}: Status Report")
        else:
            print(f"{filename}: Incorrect structure, skip")
    except Exception as e:
        print(f"Error when reading file {filename}: {e}")

# Merge data from each group
df_tx = pd.concat(transaction_lists, ignore_index=True).sort_values(by='Date') if transaction_lists else pd.DataFrame()
df_status = pd.concat(static_status_list, ignore_index=True).drop_duplicates() if static_status_list else pd.DataFrame()

if not df_status.empty:
    df_status['Norm_Name'] = df_status['Player'].apply(normalize_name)


# Part III: Filter season
all_healthy_seasons = []
unique_years = sorted(df_stats_clean['Season_Year'].unique().astype(int))

print("Running injury matching algorithm")

for year in unique_years:
    injured_set = set()
    
    # Option A: Scanning using the timeline algorithm (Log Group)
    if not df_tx.empty:
        start_date, playoff_start = get_season_dates(year)
        season_mask = (df_tx['Date'] >= start_date) & (df_tx['Date'] <= playoff_start)
        df_season_tx = df_tx[season_mask]
        
        for _, row in df_season_tx.iterrows():
            for p in extract_player_names(row['Relinquished']):
                injured_set.add(normalize_name(p))
            for p in extract_player_names(row['Acquired']):
                norm_p = normalize_name(p)
                if norm_p in injured_set:
                    injured_set.remove(norm_p)
                    
    # Option B: Directly add static injury cases (SportsRef Group)
    if not df_status.empty:
        status_current_year = df_status[df_status['Season_Year'] == year]
        for p in status_current_year['Norm_Name']:
            injured_set.add(p)
            
    # Perform an Anti-Join to filter out injured players of the current year
    df_stats_year = df_stats_clean[df_stats_clean['Season_Year'] == year].copy()
    total_players_year = len(df_stats_year)
    
    df_healthy_year = df_stats_year[~df_stats_year['Norm_Name'].isin(injured_set)].copy()
    removed_count = total_players_year - len(df_healthy_year)
    
    all_healthy_seasons.append(df_healthy_year)
    print(f"Season {year}: Previous {total_players_year} -> Remove {removed_count} -> Remaining {len(df_healthy_year)} players")


# Part IV: Aggregate and export the final data file
if all_healthy_seasons:
    combined_healthy_data = pd.concat(all_healthy_seasons, ignore_index=True)
    combined_healthy_data.drop(columns=['Norm_Name'], inplace=True, errors='ignore')
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/interim/player_data_playoff_ready.csv'
    combined_healthy_data.to_csv(out_path, index=False)
    
    print(f"Total number of healthy players available for the Play-offs (2000-2025): {len(combined_healthy_data)}")
    print(f"The clean data file has been saved at: '{out_path}'")
else:
    print("No data was exported")