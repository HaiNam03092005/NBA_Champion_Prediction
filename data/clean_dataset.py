import pandas as pd
import os

# The path for raw dataset
folder_path = 'data/nba_history'

# Create empty list for containing the data
all_seasons_list = []

# Run each file in nba_history
for file_name in os.listdir(folder_path):
    
    # Run the files with tail '.xls'
    if file_name.endswith('.xls'):
        
        # Select years from files (example: '2015.xls' -> 2015)
        season_year = int(file_name.split('.')[0])
        file_path = os.path.join(folder_path, file_name)
        
        # Read HTML file
        df_season = pd.read_html(file_path)[0]
        
        #  Flatten MultiIndex
        if isinstance(df_season.columns, pd.MultiIndex):
            new_cols = []
            for col in df_season.columns:
                level_0, level_1 = str(col[0]), str(col[1])
                # If above floor is nothing/unnamed, just choose the below floor
                if 'Unnamed' in level_0:
                    new_cols.append(level_1)
                else:
                    # Merge two floors by the '_'
                    new_cols.append(f"{level_0}_{level_1}")
            df_season.columns = new_cols
        
        # Clean the empty column and the rows that do not have the effect for the dataset
        # Delete the columns still have Unamed after Flatten MultiIndex
        df_season = df_season.loc[:, ~df_season.columns.str.contains('Unnamed')]
        
        # Delete the repeat title from the basketball reference
        df_season = df_season[df_season['Team'] != 'Team'].copy()
        
        # Delete the rows League Average
        df_season = df_season[df_season['Team'] != 'League Average'].copy()
        
        # Delete the columns that are not meaningful about basketball tactic
        cols_to_drop = ['Rk', 'Arena', 'Attend.', 'Attend./G']
        df_season = df_season.drop(columns=[col for col in cols_to_drop if col in df_season.columns])
        
        # Attack Play-off for the teams of each season
        df_season['Is_Playoff'] = df_season['Team'].astype(str).str.contains(r'\*').astype(int)
        
        # Delete the '*' to return clean team (ex: 'Miami Heat*' -> 'Miami Heat')
        df_season['Team'] = df_season['Team'].astype(str).str.replace(r'\*', '', regex=True)
        
        # Attach the years of the seasons
        df_season['Season'] = season_year
        
        # Add to the overall list
        all_seasons_list.append(df_season)
    
    # Add the champions of the test seasons    
    champions_history = {
    2010: 'Los Angeles Lakers', 2011: 'Dallas Mavericks', 2012: 'Miami Heat',
    2013: 'Miami Heat', 2014: 'San Antonio Spurs', 2015: 'Golden State Warriors',
    2016: 'Cleveland Cavaliers', 2017: 'Golden State Warriors', 2018: 'Golden State Warriors',
    2019: 'Toronto Raptors', 2020: 'Los Angeles Lakers', 2021: 'Milwaukee Bucks',
    2022: 'Golden State Warriors', 2023: 'Denver Nuggets', 2024: 'Boston Celtics'
}
    # Add the runners_up of the test seasons 
    runners_up_history = {
    2010: 'Boston Celtics', 2011: 'Miami Heat', 2012: 'Oklahoma City Thunder',
    2013: 'San Antonio Spurs', 2014: 'Miami Heat', 2015: 'Cleveland Cavaliers',
    2016: 'Golden State Warriors', 2017: 'Cleveland Cavaliers', 2018: 'Cleveland Cavaliers',
    2019: 'Golden State Warriors', 2020: 'Miami Heat', 2021: 'Phoenix Suns',
    2022: 'Boston Celtics', 2023: 'Miami Heat', 2024: 'Dallas Mavericks'
}
    
   # Check the Target (4 condition: 0, 1, 2, 3)
    def assign_target(row):
        year = row['Season']
        team = row['Team']
    
        if year in champions_history and champions_history[year] == team:
            return 3
        elif year in runners_up_history and runners_up_history[year] == team:
            return 2
        elif row['Is_Playoff'] == 1:
            return 1
        else:
            return 0
    

# Connecting any list into only one DataFrame
if len(all_seasons_list) > 0:
    master_df = pd.concat(all_seasons_list, ignore_index=True)
    
    # Apply the 'Target' column
    master_df['Target'] = master_df.apply(assign_target, axis=1)
    
    
    # Save the output to one file csv in data folder
    master_df.to_csv('data/master_dataset.csv', index=False)
else:
    print(" Empty folder, can't find anything file '.xls'!")