import os
from supabase_client import supabase
import pandas as pd

def check_counts():
    """Checks the row counts of the tables in the Supabase database."""
    print("Checking row counts in Supabase tables...")
    try:
        # Check teams count
        teams_count_response = supabase.table('teams').select('id', count='exact').execute()
        print(f"Found {teams_count_response.count} rows in 'teams' table.")

        # Check games count
        games_count_response = supabase.table('games').select('game_id', count='exact').execute()
        print(f"Found {games_count_response.count} rows in 'games' table.")

        # Check team_game_logs count
        logs_count_response = supabase.table('team_game_logs').select('game_id', count='exact').execute()
        print(f"Found {logs_count_response.count} rows in 'team_game_logs' table.")

    except Exception as e:
        print(f"An error occurred while checking table counts: {e}")

if __name__ == "__main__":
    check_counts()
