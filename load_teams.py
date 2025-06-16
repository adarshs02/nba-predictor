#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A dedicated script to fetch all NBA teams and populate the Supabase 'teams' table."""

import os
import sys
from dotenv import load_dotenv
from nba_api.stats.static import teams
from supabase_client import supabase

load_dotenv()

def populate_teams_table():
    """
    Fetches all NBA teams from the API and populates the 'teams' table in Supabase.
    """
    print("Fetching all teams from NBA API...")
    try:
        all_teams = teams.get_teams()
        if not all_teams:
            print("No teams found from the API.", file=sys.stderr)
            return

        print(f"Found {len(all_teams)} teams. Formatting for database insertion...")

        team_records = []
        for team_info in all_teams:
            team_record = {
                'id': team_info.get('id'),
                'full_name': team_info.get('full_name'),
                'abbreviation': team_info.get('abbreviation'),
                'nickname': team_info.get('nickname'),
                'city': team_info.get('city'),
                'state': team_info.get('state'),
                'year_founded': team_info.get('year_founded'),
                'logo_url': f"https://cdn.nba.com/logos/nba/{team_info.get('id')}/global/L/logo.svg"
            }
            team_records.append(team_record)

        print(f"Upserting {len(team_records)} team records into the 'teams' table...")
        response = supabase.table('teams').upsert(team_records, on_conflict='id').execute()

        # Check for errors in the response
        if hasattr(response, 'error') and response.error:
             print(f"Error inserting teams: {response.error}", file=sys.stderr)
        else:
            print("Successfully populated the 'teams' table.")

    except Exception as e:
        print(f"An error occurred during team population: {e}", file=sys.stderr)

if __name__ == "__main__":
    populate_teams_table()
