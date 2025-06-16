-- Supabase Schema for NBA Predictor
-- Run this in your Supabase project's SQL Editor to create the necessary tables.

-- Create the 'teams' table
CREATE TABLE IF NOT EXISTS public.teams (
    id BIGINT PRIMARY KEY,
    full_name TEXT UNIQUE,
    abbreviation TEXT,
    nickname TEXT,
    city TEXT,
    state TEXT,
    year_founded BIGINT,
    logo_url TEXT
);

-- Create the 'games' table
CREATE TABLE IF NOT EXISTS public.games (
    game_id TEXT PRIMARY KEY,
    game_date DATE,
    home_team_id BIGINT REFERENCES public.teams(id),
    away_team_id BIGINT REFERENCES public.teams(id),
    home_team_score BIGINT,
    away_team_score BIGINT
);

-- Create the 'team_game_logs' table
CREATE TABLE IF NOT EXISTS public.team_game_logs (
    game_id TEXT,
    team_id BIGINT,
    game_date DATE,
    matchup TEXT,
    wl TEXT,
    min BIGINT,
    fgm BIGINT,
    fga BIGINT,
    fg_pct REAL,
    fg3m BIGINT,
    fg3a BIGINT,
    fg3_pct REAL,
    ftm BIGINT,
    fta BIGINT,
    ft_pct REAL,
    oreb BIGINT,
    dreb BIGINT,
    reb BIGINT,
    ast BIGINT,
    stl BIGINT,
    blk BIGINT,
    tov BIGINT,
    pf BIGINT,
    pts BIGINT,
    plus_minus BIGINT,
    season_year TEXT,
    PRIMARY KEY (game_id, team_id)
);

-- Add comments to tables for clarity
COMMENT ON TABLE public.teams IS 'Stores static information about each NBA team.';
COMMENT ON TABLE public.games IS 'Stores game-level data, including matchups and final scores.';
COMMENT ON TABLE public.team_game_logs IS 'Stores detailed statistical logs for each team in each game.';
