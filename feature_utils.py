import pandas as pd
import numpy as np
from datetime import datetime, timedelta # Ensure timedelta is imported if not already via pandas

def calculate_team_rolling_stats(team_id, game_date, team_logs_df, window_size=10):
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])
    game_date = pd.to_datetime(game_date)
    past_games = team_logs_df[(team_logs_df['team_id'] == team_id) & (team_logs_df['game_date'] < game_date)].sort_values(by='game_date', ascending=False)
    
    # DEBUG: Inside calculate_team_rolling_stats
    print(f"    [Debug calculate_team_rolling_stats] Team ID: {team_id}, Game Date: {game_date.strftime('%Y-%m-%d')}")
    print(f"    [Debug calculate_team_rolling_stats] past_games shape: {past_games.shape}")
    if not past_games.empty:
        print(f"    [Debug calculate_team_rolling_stats] past_games date range: {past_games['game_date'].min().strftime('%Y-%m-%d')} to {past_games['game_date'].max().strftime('%Y-%m-%d')}")

    if past_games.empty or len(past_games) < 3: # Minimum 3 games to calculate meaningful stats
        print(f"    [Debug calculate_team_rolling_stats] Condition met: past_games empty or len < 3. Returning all NaNs.")
        # Return a Series with NaN values for all expected stats to maintain structure
        expected_stats_cols = [
            'avg_pts', 'avg_fgm', 'avg_fga', 'avg_fg_pct', 'avg_fg3m', 'avg_fg3a', 'avg_fg3_pct', 
            'avg_ftm', 'avg_fta', 'avg_ft_pct', 'avg_oreb', 'avg_dreb', 'avg_reb', 'avg_ast', 
            'avg_stl', 'avg_blk', 'avg_tov', 'avg_pf', 'avg_plus_minus', 'avg_win_pct',
            'offensive_rating', 'defensive_rating', 'efficiency_rating' #, 'season_phase', 'days_since_last_game' # these seem out of place here
        ]
        return pd.Series(index=expected_stats_cols, dtype=float)

    past_games_10 = past_games.head(min(window_size, len(past_games))) # Use window_size consistently
    print(f"    [Debug calculate_team_rolling_stats] past_games_10 shape: {past_games_10.shape}")
    if not past_games_10.empty:
        print(f"    [Debug calculate_team_rolling_stats] past_games_10 date range: {past_games_10['game_date'].min().strftime('%Y-%m-%d')} to {past_games_10['game_date'].max().strftime('%Y-%m-%d')}")

    team_stats = {}
    basic_stats = ['pts', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 
                   'ftm', 'fta', 'ft_pct', 'oreb', 'dreb', 'reb', 'ast', 
                   'stl', 'blk', 'tov', 'pf', 'plus_minus']
    
    # Ensure numeric conversion for basic_stats
    for col in basic_stats:
        if col in past_games_10.columns:
            # Use .loc with .copy() earlier or ensure past_games_10 is a copy
            # For safety, let's operate on a copy for modifications
            past_games_10_copy = past_games_10.copy()
            past_games_10_copy.loc[:, col] = pd.to_numeric(past_games_10_copy[col], errors='coerce')
            team_stats[f'avg_{col}'] = past_games_10_copy[col].mean()
        else:
            team_stats[f'avg_{col}'] = np.nan # Ensure key exists

    if 'wl' in past_games_10.columns:
        team_stats['avg_win_pct'] = (past_games_10['wl'] == 'W').mean()
    else:
        team_stats['avg_win_pct'] = np.nan

    # ADVANCED METRICS
    avg_fga = team_stats.get('avg_fga', np.nan)
    avg_oreb = team_stats.get('avg_oreb', np.nan)
    avg_tov = team_stats.get('avg_tov', np.nan)
    avg_pts = team_stats.get('avg_pts', np.nan)
    avg_fta = team_stats.get('avg_fta', np.nan)

    if not np.isnan(avg_fga) and not np.isnan(avg_oreb) and not np.isnan(avg_tov) and not np.isnan(avg_pts) and not np.isnan(avg_fta):
        possessions = avg_fga - avg_oreb + avg_tov + (0.44 * avg_fta)
        if possessions > 0:
            team_stats['offensive_rating'] = (avg_pts / possessions) * 100
        else:
            team_stats['offensive_rating'] = np.nan
    else:
        team_stats['offensive_rating'] = np.nan
    
    if 'opp_pts' in past_games_10.columns and 'opp_fga' in past_games_10.columns and 'opp_fta' in past_games_10.columns and 'opp_tov' in past_games_10.columns and 'opp_oreb' in past_games_10.columns:
        past_games_10_copy = past_games_10.copy() # Work on a copy
        past_games_10_copy.loc[:, 'opp_pts'] = pd.to_numeric(past_games_10_copy['opp_pts'], errors='coerce')
        past_games_10_copy.loc[:, 'opp_fga'] = pd.to_numeric(past_games_10_copy['opp_fga'], errors='coerce')
        past_games_10_copy.loc[:, 'opp_fta'] = pd.to_numeric(past_games_10_copy['opp_fta'], errors='coerce')
        past_games_10_copy.loc[:, 'opp_tov'] = pd.to_numeric(past_games_10_copy['opp_tov'], errors='coerce')
        past_games_10_copy.loc[:, 'opp_oreb'] = pd.to_numeric(past_games_10_copy['opp_oreb'], errors='coerce')
        
        opp_possessions = past_games_10_copy['opp_fga'] - past_games_10_copy['opp_oreb'] + past_games_10_copy['opp_tov'] + (0.44 * past_games_10_copy['opp_fta'])
        valid_opp_poss_indices = opp_possessions[opp_possessions > 0].index
        if not valid_opp_poss_indices.empty:
            team_stats['defensive_rating'] = (past_games_10_copy.loc[valid_opp_poss_indices, 'opp_pts'] / opp_possessions[valid_opp_poss_indices] * 100).mean()
        else:
            team_stats['defensive_rating'] = np.nan
    else:
        team_stats['defensive_rating'] = np.nan
    
    avg_ast = team_stats.get('avg_ast', np.nan)
    avg_stl = team_stats.get('avg_stl', np.nan)
    if not np.isnan(avg_pts) and not np.isnan(avg_tov) and not np.isnan(avg_ast) and not np.isnan(avg_stl):
        if avg_tov > 0:
             team_stats['efficiency_rating'] = (avg_pts + avg_ast + avg_stl) / avg_tov
        else:
            team_stats['efficiency_rating'] = np.nan
    else:
        team_stats['efficiency_rating'] = np.nan
        
    # DEBUG: Final team_stats before returning Series
    print(f"    [Debug calculate_team_rolling_stats] Final team_stats dictionary for Team ID {team_id}: {{key_count: {len(team_stats)}, sample_keys: {list(team_stats.keys())[:5]}}}") # Avoid printing huge dict
    # print(f"    [Debug calculate_team_rolling_stats] Full team_stats: {team_stats}") # Uncomment for very detailed view if needed
    return pd.Series(team_stats)

def create_head_to_head_features(home_team_id, away_team_id, game_date, team_logs_df, lookback_years=3):
    game_date = pd.to_datetime(game_date)
    team_logs_df = team_logs_df.copy()
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])
    
    lookback_date = game_date - pd.Timedelta(days=365*lookback_years)
    
    home_games_mask = (
        (team_logs_df['team_id'] == home_team_id) &
        (team_logs_df['matchup'].str.contains(str(away_team_id), na=False)) &
        (team_logs_df['game_date'] < game_date) &
        (team_logs_df['game_date'] >= lookback_date)
    )
    home_games = team_logs_df[home_games_mask].sort_values(by='game_date', ascending=False)
    
    away_games_mask = (
        (team_logs_df['team_id'] == away_team_id) &
        (team_logs_df['matchup'].str.contains(str(home_team_id), na=False)) &
        (team_logs_df['game_date'] < game_date) &
        (team_logs_df['game_date'] >= lookback_date)
    )
    away_games = team_logs_df[away_games_mask].sort_values(by='game_date', ascending=False)
    
    h2h_feature_names = [
        'h2h_home_win_pct', 'h2h_home_win_streak', 'h2h_home_lose_streak', 
        'h2h_last5_win_pct', 'h2h_weighted_win_pct', 'h2h_last_game',
        'h2h_avg_diff_pts', 'h2h_trend_diff_pts', 'h2h_std_diff_pts', 'h2h_last_diff_pts',
        'h2h_avg_diff_fg_pct', 'h2h_trend_diff_fg_pct', 'h2h_std_diff_fg_pct', 'h2h_last_diff_fg_pct',
        'h2h_avg_diff_fg3_pct', 'h2h_trend_diff_fg3_pct', 'h2h_std_diff_fg3_pct', 'h2h_last_diff_fg3_pct',
        'h2h_avg_diff_ft_pct', 'h2h_trend_diff_ft_pct', 'h2h_std_diff_ft_pct', 'h2h_last_diff_ft_pct',
        'h2h_avg_diff_reb', 'h2h_trend_diff_reb', 'h2h_std_diff_reb', 'h2h_last_diff_reb',
        'h2h_avg_diff_ast', 'h2h_trend_diff_ast', 'h2h_std_diff_ast', 'h2h_last_diff_ast',
        'h2h_avg_diff_stl', 'h2h_trend_diff_stl', 'h2h_std_diff_stl', 'h2h_last_diff_stl',
        'h2h_avg_diff_blk', 'h2h_trend_diff_blk', 'h2h_std_diff_blk', 'h2h_last_diff_blk',
        'h2h_avg_diff_tov', 'h2h_trend_diff_tov', 'h2h_std_diff_tov', 'h2h_last_diff_tov',
        'h2h_avg_diff_plus_minus', 'h2h_trend_diff_plus_minus', 'h2h_std_diff_plus_minus', 'h2h_last_diff_plus_minus',
        'h2h_composite_score', 'h2h_home_court_advantage'
    ]
    features = {name: np.nan for name in h2h_feature_names}

    if home_games.empty or away_games.empty:
        return features

    home_games_renamed = home_games.rename(columns={
        'pts': 'h_pts', 'wl': 'h_wl', 'fg_pct': 'h_fg_pct', 'fg3_pct': 'h_fg3_pct', 
        'ft_pct': 'h_ft_pct', 'reb': 'h_reb', 'ast': 'h_ast', 'stl': 'h_stl', 
        'blk': 'h_blk', 'tov': 'h_tov', 'plus_minus': 'h_plus_minus',
        'opp_pts': 'a_pts_as_opp', 'opp_fg_pct': 'a_fg_pct_as_opp', 'opp_fg3_pct': 'a_fg3_pct_as_opp',
        'opp_ft_pct': 'a_ft_pct_as_opp', 'opp_reb': 'a_reb_as_opp', 'opp_ast': 'a_ast_as_opp',
        'opp_stl': 'a_stl_as_opp', 'opp_blk': 'a_blk_as_opp', 'opp_tov': 'a_tov_as_opp'
    })
    
    away_games_renamed = away_games.rename(columns={
        'pts': 'a_pts', 'wl': 'a_wl', 'fg_pct': 'a_fg_pct', 'fg3_pct': 'a_fg3_pct', 
        'ft_pct': 'a_ft_pct', 'reb': 'a_reb', 'ast': 'a_ast', 'stl': 'a_stl', 
        'blk': 'a_blk', 'tov': 'a_tov', 'plus_minus': 'a_plus_minus',
        'opp_pts': 'h_pts_as_opp', 'opp_fg_pct': 'h_fg_pct_as_opp', 'opp_fg3_pct': 'h_fg3_pct_as_opp',
        'opp_ft_pct': 'h_ft_pct_as_opp', 'opp_reb': 'h_reb_as_opp', 'opp_ast': 'h_ast_as_opp',
        'opp_stl': 'h_stl_as_opp', 'opp_blk': 'h_blk_as_opp', 'opp_tov': 'h_tov_as_opp'
    })

    common_game_ids = set(home_games_renamed['game_id']).intersection(set(away_games_renamed['game_id']))

    if not common_game_ids:
        return features

    h_games_common = home_games_renamed[home_games_renamed['game_id'].isin(common_game_ids)].set_index('game_id')
    h_games_common = h_games_common.sort_values('game_date', ascending=False) # Sort before using iloc
    a_games_common = away_games_renamed[away_games_renamed['game_id'].isin(common_game_ids)].set_index('game_id')
    # a_games_common should also be sorted if its order matters for direct stat comparison, though merge handles alignment

    if 'h_wl' in h_games_common.columns:
        features['h2h_home_win_pct'] = (h_games_common['h_wl'] == 'W').mean()
        
        win_streak = 0
        wl_series = h_games_common['h_wl'].values # Already sorted
        for wl_val in wl_series:
            if wl_val == 'W': win_streak += 1
            else: break
        features['h2h_home_win_streak'] = win_streak
        
        lose_streak = 0
        for wl_val in wl_series:
            if wl_val == 'L': lose_streak += 1
            else: break
        features['h2h_home_lose_streak'] = lose_streak

        last_n = min(5, len(h_games_common))
        if last_n > 0:
            features['h2h_last5_win_pct'] = (h_games_common.head(last_n)['h_wl'] == 'W').mean()
            if last_n >= 1:
                 features['h2h_last_game'] = 1 if h_games_common.iloc[0]['h_wl'] == 'W' else 0

            if last_n >= 3:
                weights = np.array([0.5, 0.3, 0.2, 0.1, 0.05])[:last_n]
                weights = weights / weights.sum()
                win_indicators = (h_games_common.head(last_n)['h_wl'] == 'W').astype(int).values
                features['h2h_weighted_win_pct'] = np.sum(win_indicators * weights)
    
    h_stat_cols_map = {
        'h_pts': 'h2h_home_pts', 'h_fg_pct': 'h2h_home_fg_pct', 'h_fg3_pct': 'h2h_home_fg3_pct',
        'h_ft_pct': 'h2h_home_ft_pct', 'h_reb': 'h2h_home_reb', 'h_ast': 'h2h_home_ast',
        'h_stl': 'h2h_home_stl', 'h_blk': 'h2h_home_blk', 'h_tov': 'h2h_home_tov',
        'h_plus_minus': 'h2h_home_plus_minus'
    }
    # For away team, we need their direct stats from their perspective in these H2H games
    a_stat_cols_map = {
        'a_pts': 'h2h_away_pts', 'a_fg_pct': 'h2h_away_fg_pct', 'a_fg3_pct': 'h2h_away_fg3_pct',
        'a_ft_pct': 'h2h_away_ft_pct', 'a_reb': 'h2h_away_reb', 'a_ast': 'h2h_away_ast',
        'a_stl': 'h2h_away_stl', 'a_blk': 'h2h_away_blk', 'a_tov': 'h2h_away_tov',
        'a_plus_minus': 'h2h_away_plus_minus'
    }

    h_df_to_merge = h_games_common[list(h_stat_cols_map.keys())].rename(columns=h_stat_cols_map)
    a_df_to_merge = a_games_common[list(a_stat_cols_map.keys())].rename(columns=a_stat_cols_map)
    
    # Before merging, ensure 'game_date' is available for sorting if it was in the index
    # h_df_to_merge['game_date'] = h_games_common['game_date'] # if game_date is not already a col
    # a_df_to_merge['game_date'] = a_games_common['game_date']

    merged_h2h_stats = pd.merge(h_df_to_merge, a_df_to_merge, left_index=True, right_index=True, how='inner') # Merge on game_id index
    # merged_h2h_stats = merged_h2h_stats.sort_values(by='game_date', ascending=False) # if game_date was added as a column
    # If game_date was part of h_games_common index, it should carry over or be re-added for sorting. Assuming h_games_common was already sorted. 

    if not merged_h2h_stats.empty:
        base_stats_for_diff = ['pts', 'fg_pct', 'fg3_pct', 'ft_pct', 'reb', 'ast', 'stl', 'blk', 'tov', 'plus_minus']
        
        for stat_base_name in base_stats_for_diff:
            home_col = f'h2h_home_{stat_base_name}'
            away_col = f'h2h_away_{stat_base_name}'
            diff_col_name_in_df = f'h2h_merged_diff_{stat_base_name}' # temp name for column in df

            if home_col in merged_h2h_stats.columns and away_col in merged_h2h_stats.columns:
                merged_h2h_stats[home_col] = pd.to_numeric(merged_h2h_stats[home_col], errors='coerce')
                merged_h2h_stats[away_col] = pd.to_numeric(merged_h2h_stats[away_col], errors='coerce')
                
                merged_h2h_stats[diff_col_name_in_df] = merged_h2h_stats[home_col] - merged_h2h_stats[away_col]
                
                features[f'h2h_avg_diff_{stat_base_name}'] = merged_h2h_stats[diff_col_name_in_df].mean()
                features[f'h2h_last_diff_{stat_base_name}'] = merged_h2h_stats[diff_col_name_in_df].iloc[0] if not merged_h2h_stats[diff_col_name_in_df].empty else np.nan
                
                if len(merged_h2h_stats) >= 3:
                    features[f'h2h_std_diff_{stat_base_name}'] = merged_h2h_stats[diff_col_name_in_df].std()
                    recent_mean = merged_h2h_stats[diff_col_name_in_df].iloc[:2].mean()
                    previous_mean = merged_h2h_stats[diff_col_name_in_df].iloc[2:].mean()
                    if not np.isnan(recent_mean) and not np.isnan(previous_mean):
                         features[f'h2h_trend_diff_{stat_base_name}'] = recent_mean - previous_mean
                    else:
                         features[f'h2h_trend_diff_{stat_base_name}'] = np.nan
                else:
                    features[f'h2h_std_diff_{stat_base_name}'] = np.nan
                    features[f'h2h_trend_diff_{stat_base_name}'] = np.nan
        
        avg_diff_pts = features.get('h2h_avg_diff_pts', np.nan)
        avg_diff_plus_minus = features.get('h2h_avg_diff_plus_minus', np.nan)
        if not np.isnan(avg_diff_pts) and not np.isnan(avg_diff_plus_minus):
            features['h2h_composite_score'] = (avg_diff_pts * 0.7 + avg_diff_plus_minus * 0.3)
        
    h2h_win_pct = features.get('h2h_home_win_pct', 0.5)
    features['h2h_home_court_advantage'] = h2h_win_pct - 0.5 

    return features


def calculate_rest_b2b_features(team_id, current_game_date, team_logs_df):
    current_game_date = pd.to_datetime(current_game_date)
    
    team_games_before_current = team_logs_df[
        (team_logs_df['team_id'] == team_id) & 
        (pd.to_datetime(team_logs_df['game_date']) < current_game_date)
    ].copy()
    
    team_games_before_current['game_date'] = pd.to_datetime(team_games_before_current['game_date'])
    team_games_before_current = team_games_before_current.sort_values('game_date', ascending=False)
    
    rest_days = 14 
    is_b2b = 0

    if not team_games_before_current.empty:
        last_played_date = team_games_before_current.iloc[0]['game_date']
        rest_days = (current_game_date - last_played_date).days
        rest_days = min(rest_days, 14)
        
        if rest_days == 1:
            is_b2b = 1
            
    return {'rest_days': rest_days, 'is_b2b': is_b2b}
