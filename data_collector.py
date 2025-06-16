import pandas as pd
import time
import os
import sys
import numpy as np
from datetime import date, timedelta
from nba_api.stats.endpoints import teamgamelogs, leaguegamefinder
from postgrest.exceptions import APIError

# Import the centralized Supabase client
from supabase_client import supabase

"""
Collects game and team log data from the NBA API and stores it in a Supabase PostgreSQL database.
This script assumes the 'teams' table has already been populated by 'load_teams.py'.
"""

def get_team_game_logs_for_season(team_id, season_nullable="2023-24", season_type_nullable="Regular Season"):
    """Fetches game logs, filters them to match the DB schema, and upserts them."""
    print(f"Fetching game logs for team {team_id}, season {season_nullable}...")
    try:
        schema_columns = [
            'game_id', 'team_id', 'game_date', 'matchup', 'wl', 'min', 'fgm',
            'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta',
            'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov',
            'pf', 'pts', 'plus_minus', 'season_year'
        ]
        
        # Define which columns should be integers based on the schema
        int_columns = [
            'min', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta', 'oreb', 'dreb', 
            'reb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'plus_minus'
        ]

        logs = teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id,
            season_nullable=season_nullable,
            season_type_nullable=season_type_nullable
        ).get_data_frames()[0]

        if logs.empty:
            print(f"No game logs found for team {team_id} in season {season_nullable}.")
            return

        logs.columns = [col.lower() for col in logs.columns]
        logs['season_year'] = season_nullable
        
        # Filter the DataFrame to only include columns that exist in our schema
        columns_to_keep = [col for col in schema_columns if col in logs.columns]
        logs_filtered = logs[columns_to_keep].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Clean and convert data types robustly
        for col in logs_filtered.columns:
            if col in int_columns:
                # Safely convert to numeric, coerce errors, fill NA with 0, and cast to int
                logs_filtered[col] = pd.to_numeric(logs_filtered[col], errors='coerce').fillna(0).astype(int)
            elif logs_filtered[col].dtype == 'object':
                # Replace potential numpy NaNs with None for JSON compatibility
                logs_filtered[col] = logs_filtered[col].apply(lambda x: None if pd.isna(x) else x)

        records = logs_filtered.to_dict(orient='records')
        supabase.table('team_game_logs').upsert(records, on_conflict='game_id,team_id').execute()
        print(f"Stored {len(records)} game logs for team {team_id} for season {season_nullable}.")
    except Exception as e:
        print(f"A general error occurred in get_team_game_logs_for_season for team {team_id}: {e}", file=sys.stderr)

def get_and_store_games_for_date(game_date):
    """Fetches scoreboard and stores game data. Assumes teams are already in the DB."""
    print(f"Fetching scoreboard for {game_date}...")
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=game_date,
            date_to_nullable=game_date,
            league_id_nullable='00'
        )
        games_df = gamefinder.get_data_frames()[0]

        if games_df.empty:
            return None

        processed_games = {}
        team_ids_in_games = set()

        # Group by game_id to process each game once
        for game_id, game_group in games_df.groupby('GAME_ID'):
            if len(game_group) != 2:
                continue # Skip if it's not a standard 2-team game

            team1_row = game_group.iloc[0]
            team2_row = game_group.iloc[1]

            # Determine home and away teams. The 'MATCHUP' field tells us.
            if ' vs. ' in team1_row['MATCHUP']:
                home_team_row = team1_row
                away_team_row = team2_row
            else:
                home_team_row = team2_row
                away_team_row = team1_row
            
            home_team_id = int(home_team_row['TEAM_ID'])
            away_team_id = int(away_team_row['TEAM_ID'])

            processed_games[game_id] = {
                'game_id': game_id,
                'game_date': pd.to_datetime(home_team_row['GAME_DATE']).date().isoformat(),
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_score': int(home_team_row['PTS']) if pd.notna(home_team_row['PTS']) else None,
                'away_team_score': int(away_team_row['PTS']) if pd.notna(away_team_row['PTS']) else None
            }
            team_ids_in_games.add(home_team_id)
            team_ids_in_games.add(away_team_id)

        if not processed_games:
            return None

        records = list(processed_games.values())
        supabase.table('games').upsert(records, on_conflict='game_id').execute()
        print(f"Stored {len(records)} games for {game_date}.")
        return team_ids_in_games

    except Exception as e:
        print(f"An error occurred in get_and_store_games_for_date for {game_date}: {e}", file=sys.stderr)
    
    return None

if __name__ == "__main__":
    print("Starting data collector for games and logs.")
    
    # Define the date range for the NBA season
    start_date = date(2024, 10, 18)
    end_date = date.today()
    
    print(f"\nStarting data collection from {start_date} to {end_date}...")

    current_date = start_date
    delta = timedelta(days=1)
    processed_team_season_logs = set()

    while current_date <= end_date:
        date_str_mmddyyyy = current_date.strftime("%m/%d/%Y")
        print(f"\n--- Processing data for {date_str_mmddyyyy} ---")
        
        team_ids_from_games = get_and_store_games_for_date(game_date=date_str_mmddyyyy)

        if team_ids_from_games:
            year = current_date.year
            month = current_date.month
            season_str_for_logs = f"{year-1}-{str(year)[-2:]}" if month < 9 else f"{year}-{str(year+1)[-2:]}"
            for team_id in team_ids_from_games:
                team_season_key = (team_id, season_str_for_logs)
                if team_season_key not in processed_team_season_logs:
                    get_team_game_logs_for_season(
                        team_id=team_id,
                        season_nullable=season_str_for_logs
                    )
                    processed_team_season_logs.add(team_season_key)

        current_date += delta

    print("\n--- Data collection loop finished ---")
    print("\nData collection script completed.")