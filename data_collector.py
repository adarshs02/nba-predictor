from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playercareerstats, playergamelog, teamgamelogs, leaguegamefinder
import pandas as pd
import time # Import time for rate limiting
import sqlite3
import numpy as np
import os
from dotenv import load_dotenv

"""
Collects data from the NBA API and stores it in a SQLite database.
"""

# Load environment variables from .env file
load_dotenv()

# Get DB_NAME from environment variable, with a default fallback
DB_NAME = os.getenv("DB_NAME", "nba_data.db")

def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Teams table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS teams (
        id INTEGER PRIMARY KEY,
        full_name TEXT UNIQUE,
        abbreviation TEXT,
        nickname TEXT,
        city TEXT,
        state TEXT,
        year_founded INTEGER
    )
    """)

    # Players table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS players (
        id INTEGER PRIMARY KEY,
        full_name TEXT,
        first_name TEXT,
        last_name TEXT,
        is_active BOOLEAN
    )
    """)

    # Games table (from scoreboard)
    # We'll need to see the structure of scoreboard_df to finalize columns
    # For now, a basic structure:
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        game_date DATE,
        home_team_id INTEGER,
        away_team_id INTEGER,
        home_team_score INTEGER,
        away_team_score INTEGER,
        FOREIGN KEY (home_team_id) REFERENCES teams(id),
        FOREIGN KEY (away_team_id) REFERENCES teams(id)
    )
    """)

    # Player Game Logs table
    # Columns based on typical playergamelog output, might need adjustment
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS player_game_logs (
        game_id TEXT,
        player_id INTEGER,
        team_id INTEGER,
        game_date DATE,
        matchup TEXT,
        wl TEXT,
        min INTEGER,
        fgm INTEGER,
        fga INTEGER,
        fg_pct REAL,
        fg3m INTEGER,
        fg3a INTEGER,
        fg3_pct REAL,
        ftm INTEGER,
        fta INTEGER,
        ft_pct REAL,
        oreb INTEGER,
        dreb INTEGER,
        reb INTEGER,
        ast INTEGER,
        stl INTEGER,
        blk INTEGER,
        tov INTEGER,
        pf INTEGER,
        pts INTEGER,
        plus_minus INTEGER,
        season_year TEXT, 
        PRIMARY KEY (game_id, player_id),
        FOREIGN KEY (player_id) REFERENCES players(id),
        FOREIGN KEY (team_id) REFERENCES teams(id),
        FOREIGN KEY (game_id) REFERENCES games(game_id) 
    )
    """)
    # Removed season_type for now, will add if available and necessary.
    # `season_year` added to store which season this log belongs to.

    # Team Game Logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS team_game_logs (
        game_id TEXT,
        team_id INTEGER,
        game_date DATE,
        matchup TEXT,
        wl TEXT,
        min INTEGER,
        fgm INTEGER,
        fga INTEGER,
        fg_pct REAL,
        fg3m INTEGER,
        fg3a INTEGER,
        fg3_pct REAL,
        ftm INTEGER,
        fta INTEGER,
        ft_pct REAL,
        oreb INTEGER,
        dreb INTEGER,
        reb INTEGER,
        ast INTEGER,
        stl INTEGER,
        blk INTEGER,
        tov INTEGER,
        pf INTEGER,
        pts INTEGER,
        plus_minus INTEGER,
        season_year TEXT,
        PRIMARY KEY (game_id, team_id),
        FOREIGN KEY (team_id) REFERENCES teams(id),
        FOREIGN KEY (game_id) REFERENCES games(game_id)
    )
    """)
    # Removed season_type for now, will add if available and necessary.
    # `season_year` added.

    conn.commit()
    conn.close()

def get_all_teams():
    """Fetches all NBA teams and stores them in the database."""
    nba_teams_data = teams.get_teams()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    teams_to_insert = []
    for team in nba_teams_data:
        # Assuming 'id', 'full_name', 'abbreviation', 'nickname', 'city', 'state', 'year_founded' are keys
        teams_to_insert.append((
            team['id'], team['full_name'], team['abbreviation'],
            team['nickname'], team['city'], team['state'], team['year_founded']
        ))
    
    cursor.executemany("""
    INSERT OR IGNORE INTO teams (id, full_name, abbreviation, nickname, city, state, year_founded)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, teams_to_insert)
    
    conn.commit()
    conn.close()
    
    print(f"Stored/updated {len(teams_to_insert)} teams in the database.")
    return nba_teams_data # Still return for immediate use if needed

def get_all_active_players():
    """Fetches all active NBA players and their IDs and stores them in the database."""
    all_nba_players_data = players.get_active_players()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    players_to_insert = []
    for player in all_nba_players_data:
        players_to_insert.append((
            player['id'], player['full_name'], player['first_name'],
            player['last_name'], player['is_active']
        ))

    cursor.executemany("""
    INSERT OR IGNORE INTO players (id, full_name, first_name, last_name, is_active)
    VALUES (?, ?, ?, ?, ?)
    """, players_to_insert)
    
    conn.commit()
    conn.close()

    print(f"Stored/updated {len(players_to_insert)} active players in the database.")
    return all_nba_players_data # Still return for immediate use

def get_player_game_logs(player_id, season_nullable="2023-24", season_type_nullable="Regular Season"):
    """Fetches game logs for a specific player and season and stores them in the database."""
    print(f"Fetching game logs for player ID {player_id} for {season_nullable} {season_type_nullable}...")
    try:
        gamelogs_endpoint = playergamelog.PlayerGameLog(player_id=player_id, season=season_nullable, season_type_all_star=season_type_nullable)
        data_frames = gamelogs_endpoint.get_data_frames()

        if not data_frames:
            print(f"No dataframes returned from PlayerGameLog for player {player_id}, season {season_nullable}.")
            return pd.DataFrame()
        
        df = data_frames[0]

        if df is None or df.empty:
            print(f"Player game log DataFrame is None or empty for player {player_id}, season {season_nullable}.")
            return pd.DataFrame()

        actual_columns = df.columns.tolist()
        print(f"Initial Player Game Log Columns for player {player_id}: {actual_columns}")

        game_id_col_name = None
        for potential_name in ['GAME_ID', 'Game_ID', 'game_id']:
            if potential_name in actual_columns:
                game_id_col_name = potential_name
                break
        
        if not game_id_col_name:
            print(f"CRITICAL: Game ID column not found in player game logs for player {player_id}. Available: {actual_columns}")
            return pd.DataFrame()
        
        if game_id_col_name != 'GAME_ID':
            print(f"Found game ID column as '{game_id_col_name}', renaming to 'GAME_ID' for player {player_id}.")
            df.rename(columns={game_id_col_name: 'GAME_ID'}, inplace=True)
        
        df['GAME_ID'] = df['GAME_ID'].astype(str)
        df['SEASON_YEAR'] = season_nullable
        
        # Refresh actual_columns if renaming occurred
        actual_columns = df.columns.tolist() 

        db_columns_player_logs = [
            'GAME_ID', 'PLAYER_ID', 'TEAM_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
            'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
            'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
            'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 'SEASON_YEAR'
        ]

        # Select only the columns that exist in df and are needed for the database
        cols_to_insert_in_df = [col for col in db_columns_player_logs if col in actual_columns]
        df_to_insert = df[cols_to_insert_in_df]

        if not df_to_insert.empty:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            column_names_for_sql = ', '.join(df_to_insert.columns)
            placeholders = ', '.join(['?'] * len(df_to_insert.columns))
            
            for row_tuple in df_to_insert.itertuples(index=False):
                sql = f"INSERT OR REPLACE INTO player_game_logs ({column_names_for_sql}) VALUES ({placeholders})"
                cursor.execute(sql, row_tuple)
            conn.commit()
            conn.close()
            print(f"Stored/updated {len(df_to_insert)} game logs for player {player_id} for season {season_nullable}.")
        else:
            print(f"No data to insert into DB for player game logs for player {player_id} after column filtering.")
        
        return df # Return original df (or the one with renamed GAME_ID) for consistency, even if insertion part changes
    
    except Exception as e:
        print(f"Error fetching/storing game logs for player {player_id}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_team_game_logs_for_season(team_id, season_nullable="2023-24", season_type_nullable="Regular Season"):
    """Fetches game logs for a specific team and season and stores them in the database."""
    print(f"Fetching game logs for team ID {team_id} for {season_nullable} {season_type_nullable}...")
    try:
        gamelogs_endpoint = teamgamelogs.TeamGameLogs(team_id_nullable=team_id, season_nullable=season_nullable, season_type_nullable=season_type_nullable)
        df = gamelogs_endpoint.get_data_frames()[0]

        if not df.empty:
            df['GAME_ID'] = df['GAME_ID'].astype(str)
            df['SEASON_YEAR'] = season_nullable
            
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            # Ensure column names match the table definition
            log_columns = ['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
                           'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                           'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                           'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 'SEASON_YEAR']
            df_to_insert = df[log_columns]

            for row in df_to_insert.itertuples(index=False):
                placeholders = ', '.join(['?'] * len(row))
                column_names = ', '.join(df_to_insert.columns)
                sql = f"INSERT OR REPLACE INTO team_game_logs ({column_names}) VALUES ({placeholders})"
                cursor.execute(sql, row)
            conn.commit()
            conn.close()
            print(f"Stored/updated {len(df)} game logs for team {team_id} for season {season_nullable}.")
        else:
            print(f"No game logs found for team {team_id} for the specified season.")
        return df
    except Exception as e:
        print(f"Error fetching/storing game logs for team {team_id}: {e}")
        return pd.DataFrame()

def get_scoreboard_for_date(game_date="03/15/2024"):
    """Fetches scoreboard data for a specific date using LeagueGameFinder and stores it in the games table."""
    print(f"Fetching scoreboard for date: {game_date} (MM/DD/YYYY) using LeagueGameFinder...")
    try:
        # Use LeagueGameFinder to get games for a specific date.
        # Ensure game_date is in MM/DD/YYYY format as expected by date_from_nullable/date_to_nullable.
        gamefinder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=game_date, 
            date_to_nullable=game_date,
            league_id_nullable='00', # NBA
            season_type_nullable='Regular Season' # Or 'Playoffs', 'Pre Season', etc. as needed
        )
        all_games_df = gamefinder.get_data_frames()[0]

        if all_games_df.empty:
            print(f"No games found by LeagueGameFinder for date: {game_date}.")
            return pd.DataFrame()

        # LeagueGameFinder returns one row per team per game. We need to consolidate.
        # Columns typically include: TEAM_ID, TEAM_ABBREVIATION, GAME_ID, GAME_DATE, MATCHUP, WL, PTS
        
        # Filter out potential duplicate entries if any based on GAME_ID and TEAM_ID
        all_games_df = all_games_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])

        processed_games = []
        for game_id, group in all_games_df.groupby('GAME_ID'):
            if len(group) != 2:
                print(f"Warning: Did not find exactly two teams for GAME_ID {game_id} on {game_date}. Found {len(group)}. Skipping this game.")
                continue

            team1_row = group.iloc[0]
            team2_row = group.iloc[1]

            # Determine home and away team from MATCHUP string
            # Example: "DEN @ LAL" -> LAL is home. "DEN vs. PHX" -> DEN is home.
            matchup_str = team1_row['MATCHUP'] # Matchup string is the same for both rows of a game
            home_team_abbr = None
            away_team_abbr = None

            if "@" in matchup_str: # Away @ Home
                away_team_abbr, home_team_abbr = [abbr.strip() for abbr in matchup_str.split('@')]
            elif "vs." in matchup_str: # Home vs. Away
                home_team_abbr, away_team_abbr = [abbr.strip() for abbr in matchup_str.split('vs.')]
            else:
                print(f"Warning: Could not determine home/away from MATCHUP '{matchup_str}' for GAME_ID {game_id}. Skipping.")
                continue
            
            # Assign details based on determined home/away abbreviations
            if team1_row['TEAM_ABBREVIATION'] == home_team_abbr:
                home_team_data = team1_row
                away_team_data = team2_row
            elif team2_row['TEAM_ABBREVIATION'] == home_team_abbr:
                home_team_data = team2_row
                away_team_data = team1_row
            else:
                print(f"Warning: Could not match team abbreviations ('{team1_row['TEAM_ABBREVIATION']}', '{team2_row['TEAM_ABBREVIATION']}') with parsed home/away ('{home_team_abbr}', '{away_team_abbr}') for GAME_ID {game_id}. Skipping.")
                continue
            
            # GAME_DATE is in YYYY-MM-DD format from LeagueGameFinder results
            parsed_game_date = pd.to_datetime(home_team_data['GAME_DATE']).strftime('%Y-%m-%d')

            processed_games.append({
                'game_id': str(game_id),
                'game_date': parsed_game_date,
                'home_team_id': home_team_data['TEAM_ID'],
                'away_team_id': away_team_data['TEAM_ID'],
                'home_team_score': home_team_data['PTS'],
                'away_team_score': away_team_data['PTS']
            })

        if not processed_games:
            print(f"No games could be processed from LeagueGameFinder for date: {game_date}.")
            return pd.DataFrame() # Return empty if no games are processed

        games_to_insert_df = pd.DataFrame(processed_games)

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        insert_tuples = [tuple(x) for x in games_to_insert_df.to_numpy()]
        
        cursor.executemany("""
        INSERT OR REPLACE INTO games (game_id, game_date, home_team_id, away_team_id, home_team_score, away_team_score)
        VALUES (?, ?, ?, ?, ?, ?)
        """, insert_tuples)
        
        conn.commit()
        conn.close()
        print(f"Stored/updated {len(processed_games)} games from LeagueGameFinder for date {game_date}.")
        return games_to_insert_df # Return the processed DataFrame

    except Exception as e:
        print(f"Error fetching/storing scoreboard using LeagueGameFinder for {game_date}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_player_career_stats(player_id):
    """Fetches career statistics for a given player ID."""
    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    career_df = career.get_data_frames()[0]
    print(f"\nCareer stats for player ID {player_id}:")
    print(career_df.head())
    return career_df

if __name__ == "__main__":
    init_db() # Initialize database and tables first
    
    print("Fetching static data (teams)...")
    all_teams_static = get_all_teams() # Fetches and stores all teams
    time.sleep(1)

    from datetime import date, timedelta

    # Set dates for the 2022-2023 NBA season (October to June)
    start_date = date(2023, 10, 24) # Approximate start of 2023-24 season
    end_date = date(2024, 4, 14)   # End of playoffs for 2023-24 season

    print(f"\nStarting efficient data collection for game data and team logs from {start_date} to {end_date} (2022-2023 season)...")
    print("Note: This will take a significant amount of time due to the extended date range and API rate limiting.")

    current_date = start_date
    delta = timedelta(days=1)
    
    processed_team_season_logs = set() # Keep track of (team_id, season_str) already fetched

    while current_date <= end_date:
        date_str_mmddyyyy = current_date.strftime("%m/%d/%Y")
        print(f"\n--- Processing data for {date_str_mmddyyyy} ---")
        
        games_on_date_df = get_scoreboard_for_date(game_date=date_str_mmddyyyy)
        time.sleep(0.5) # Be respectful to the API

        if games_on_date_df is not None and not games_on_date_df.empty:
            season_year = current_date.year
            if current_date.month < 9: # NBA season typically flips around Sept/Oct. For logs, this logic should derive season like "2022-23"
                season_str_for_logs = f"{season_year - 1}-{str(season_year)[-2:]}"
            else: 
                season_str_for_logs = f"{season_year}-{str(season_year + 1)[-2:]}"
            print(f"Determined season for logs: {season_str_for_logs}")

            home_team_ids = pd.to_numeric(games_on_date_df['home_team_id'], errors='coerce').dropna().unique()
            away_team_ids = pd.to_numeric(games_on_date_df['away_team_id'], errors='coerce').dropna().unique()
            combined_ids_array = np.concatenate((home_team_ids, away_team_ids))
            team_ids_on_date = pd.unique(combined_ids_array)

            print(f"Teams playing on {date_str_mmddyyyy}: {team_ids_on_date}")

            for team_id_float in team_ids_on_date:
                if pd.notna(team_id_float):
                    team_id = int(team_id_float)
                    team_season_key = (team_id, season_str_for_logs)
                    
                    if team_season_key not in processed_team_season_logs:
                        print(f"Fetching team game logs for team ID: {team_id} for season {season_str_for_logs}")
                        get_team_game_logs_for_season(team_id=team_id, season_nullable=season_str_for_logs)
                        processed_team_season_logs.add(team_season_key)
                        time.sleep(0.5) # Respect API limits
                    else:
                        print(f"Team game logs for team ID: {team_id}, season {season_str_for_logs} already processed. Skipping.")
        else:
            print(f"No games found on {date_str_mmddyyyy} to process for team logs.")

        current_date += delta

    print("\n--- Data collection loop finished ---")

    #TODO: Make more player based predictions and data collection
    # # Update target season for sample player logs to match the collected year
    # target_season_for_player_logs = "2022-23"
    # jokic_id_static = "203999"
    # print(f"\nFetching sample player game logs for Nikola Jokic (ID: {jokic_id_static}) for season {target_season_for_player_logs}")
    # get_player_game_logs(player_id=jokic_id_static, season_nullable=target_season_for_player_logs)
    # time.sleep(1)

    print("\nData collection script completed.") 