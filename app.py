import streamlit as st
import re
import pandas as pd
import sys
import os
import json
from datetime import datetime

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predictor import get_prediction_data_for_teams
    from summary_api import generate_game_summary
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def format_stat_name(technical_name):
    """Convert technical variable names to user-friendly display names"""
    stat_mapping = {
        # Basic stats
        'pts': 'Points per Game',
        'fg_pct': 'Field Goal %',
        'fg3_pct': '3-Point %',
        'ft_pct': 'Free Throw %',
        'reb': 'Rebounds per Game',
        'ast': 'Assists per Game',
        'stl': 'Steals per Game',
        'blk': 'Blocks per Game',
        'tov': 'Turnovers per Game',
        'pf': 'Personal Fouls per Game',
        
        # Differential stats
        'diff_pts': 'Point Differential',
        'diff_fg_pct': 'Field Goal % Differential',
        'diff_fg3_pct': '3-Point % Differential',
        'diff_ft_pct': 'Free Throw % Differential',
        'diff_reb': 'Rebounding Differential',
        'diff_ast': 'Assist Differential',
        'diff_stl': 'Steal Differential',
        'diff_blk': 'Block Differential',
        'diff_tov': 'Turnover Differential',
        
        # Advanced stats
        'off_rating': 'Offensive Rating',
        'def_rating': 'Defensive Rating',
        'pace': 'Pace',
        'win_pct': 'Win Percentage',
        'recent_form': 'Recent Form',
        'home_record': 'Home Record',
        'away_record': 'Away Record',
        'rest_days': 'Rest Days',
        'b2b': 'Back-to-Back Games',
        
        # Head-to-head
        'h2h_win_pct': 'Head-to-Head Win %',
        'h2h_avg_margin': 'H2H Average Margin',
        
        # Team specific prefixes
        'home_avg_': 'Home Team ',
        'away_avg_': 'Away Team ',
        'home_': 'Home Team ',
        'away_': 'Away Team '
    }
    
    # Try exact match first
    if technical_name in stat_mapping:
        return stat_mapping[technical_name]
    
    # Handle prefixed stats
    for prefix, replacement in [('home_avg_', 'Home Team '), ('away_avg_', 'Away Team '), 
                               ('home_', 'Home Team '), ('away_', 'Away Team ')]:
        if technical_name.startswith(prefix):
            base_stat = technical_name.replace(prefix, '')
            base_display = stat_mapping.get(base_stat, base_stat.replace('_', ' ').title())
            return f"{replacement}{base_display}"
    
    # Fallback: convert underscores to spaces and title case
    return technical_name.replace('_', ' ').title()

def format_key_factors(factors_text):
    """Formats key prediction factors into visually appealing HTML cards."""
    if not factors_text or factors_text == "N/A":
        return "<p style='color: #333;'>No key factors available</p>"

    factors = factors_text.split('; ')
    
    # Icon mapping for different stats
    icon_map = {
        "Point": "🏀", "Field Goal": "🎯", "3-Point": "🎯", "Rebound": "🙌",
        "Assist": "🤝", "Turnover": "⚠️", "Win": "🏆", "Streak": "🔥",
        "Rest": "😴", "Momentum": "⚡️", "Efficiency": "⚙️"
    }

    cards_html = "<div style='display: flex; flex-wrap: wrap; gap: 15px; justify-content: center;'>"
    
    for factor in factors:
        if '(value:' in factor and 'importance:' in factor:
            parts = factor.split(' (value: ')
            if len(parts) == 2:
                factor_name = parts[0].strip()
                value_part = parts[1]
                value_str, importance_str = value_part.split(', importance: ')
                importance_str = importance_str.rstrip(')')
                display_name = format_stat_name(factor_name)

                try:
                    value = float(value_str)
                    importance = float(importance_str)
                    icon = next((i for key, i in icon_map.items() if key in display_name), "📊")
                    value_color = "#28a745" if value > 0 else "#dc3545"

                    cards_html += f"""
                    <div style="
                        background-color: #ffffff; 
                        border: 1px solid #e0e0e0; 
                        border-left: 5px solid {value_color};
                        border-radius: 8px; 
                        padding: 15px; 
                        width: 180px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
                        text-align: center; 
                        transition: transform 0.2s;">
                        <div style="font-size: 28px; margin-bottom: 8px;">{icon}</div>
                        <div style="font-size: 15px; font-weight: bold; color: #333333; margin-bottom: 5px; height: 40px;">{display_name}</div>
                        <div style="font-size: 18px; font-weight: bold; color: {value_color};">{value:.2f}</div>
                        <div style="font-size: 12px; color: #6c757d; margin-top: 5px;">Impact: {importance:.3f}</div>
                    </div>
                    """
                except ValueError:
                    continue

    cards_html += "</div>"
    return cards_html

def display_team_stats_enhanced(stats, logo_url=None):
    """Display team statistics in a formatted way using native Streamlit components."""
    if not stats:
        st.write("No statistics available")
        return

    # Display logo if available
    if logo_url:
        st.image(logo_url, width=100)

    # Define the stats to display
    stats_to_show = {
        "Points/Game": stats.get('pts', 'N/A'),
        "Field Goal %": stats.get('fg_pct', 'N/A'),
        "3-Point %": stats.get('fg3_pct', 'N/A'),
        "Assists/Game": stats.get('ast', 'N/A'),
        "Rebounds/Game": stats.get('reb', 'N/A'),
        "Turnovers/Game": stats.get('tov', 'N/A')
    }

    # Display stats using columns for layout
    cols = st.columns(2)
    for i, (label, value) in enumerate(stats_to_show.items()):
        with cols[i % 2]:
            st.metric(label=label, value=str(value))

def display_team_stats(stats):
    """Display team statistics in a formatted way"""
    if not stats:
        st.write("No statistics available")
        return
    
    # Create metrics for key stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Points/Game", f"{stats.get('pts', 'N/A')}")
        st.metric("Field Goal %", f"{stats.get('fg_pct', 'N/A')}")
        st.metric("3-Point %", f"{stats.get('fg3_pct', 'N/A')}")
    
    with col2:
        st.metric("Assists/Game", f"{stats.get('ast', 'N/A')}")
        st.metric("Rebounds/Game", f"{stats.get('reb', 'N/A')}")
        st.metric("Turnovers/Game", f"{stats.get('tov', 'N/A')}")

def display_prediction_results(prediction_data, llm_summary, home_team, away_team):
    """Display the prediction results in a formatted layout"""
    
    # Simple header using native Streamlit components
    st.markdown("---")
    st.markdown("# 🏀 GAME PREDICTION")
    st.markdown(f"## {home_team} vs {away_team}")
    st.markdown("---")
    
    # Prediction results with simple styling
    predicted_winner = prediction_data.get('predicted_winner_name', 'Unknown')
    win_probability = prediction_data.get('win_probability_percent', 0)
    
    # Winner announcement using native components
    st.success(f"🏆 PREDICTED WINNER: **{predicted_winner}**")
    st.info(f"Confidence: **{win_probability}%**")
    
    st.markdown("---")
    
    # Team comparison using native columns
    st.markdown("## 📊 Team Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🏠 {home_team}")
        home_stats = prediction_data.get('home_stats', {})
        home_win_pct = home_stats.get('win_pct', 'N/A')
        st.write(f"**Win Rate (Last 10 Games):** {home_win_pct}")
    
    with col2:
        st.markdown(f"### ✈️ {away_team}")
        away_stats = prediction_data.get('away_stats', {})
        away_win_pct = away_stats.get('win_pct', 'N/A')
        st.write(f"**Win Rate (Last 10 Games):** {away_win_pct}")
    
    st.markdown("---")
    
    # AI Analysis with simple container
    st.markdown("## 🤖 AI Analysis & Prediction")
    
    # Process the LLM summary to remove asterisks
    processed_summary = re.sub(r'\*\*(.*?):\*\*', r'**\1:**', llm_summary)
    
    # Use a simple container for the AI summary
    with st.container():
        st.write(processed_summary)
    
    st.markdown("---")
    
    # Enhanced Team Statistics Section
    st.markdown("## 📈 Detailed Team Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🏠 {home_team}")
        home_logo_url = prediction_data.get('home_team_logo_url')
        display_team_stats_enhanced(home_stats, home_logo_url)
    
    with col2:
        st.markdown(f"### ✈️ {away_team}")
        away_logo_url = prediction_data.get('away_team_logo_url')
        display_team_stats_enhanced(away_stats, away_logo_url)
    
    # Enhanced Key Factors Section
    if 'top_global_features' in prediction_data:
        st.markdown("---")
        st.markdown("## 🔑 Key Prediction Factors")
        
        features = prediction_data['top_global_features']
        if features:
            # Handle both string and dict formats for features
            if isinstance(features, str):
                st.write(features)
            elif isinstance(features, list):
                # Create simple cards using columns
                cols = st.columns(min(len(features), 3))
                for i, feature in enumerate(features[:6]):  # Show top 6 features
                    with cols[i % 3]:
                        if isinstance(feature, dict):
                            feature_name = feature.get('name', 'Unknown')
                            feature_value = feature.get('value', 0)
                            feature_importance = feature.get('importance', 0)
                            
                            # Clean up feature name for display
                            clean_name = feature_name.replace('_', ' ').title()
                            clean_name = clean_name.replace('Diff ', '').replace('Home ', '').replace('Away ', '')
                            
                            st.metric(
                                label=clean_name,
                                value=f"{feature_value:.3f}",
                                delta=f"Impact: {feature_importance:.3f}"
                            )
                        else:
                            # Handle string features
                            st.write(f"• {str(feature)}")
            else:
                st.write(str(features))

def main():
    st.title("🏀 NBA Game Predictor")
    st.markdown("---")
    
    # Create a description
    st.markdown("""
    **Welcome to the NBA Game Predictor!**
    
    Enter two NBA teams below to get an AI-powered prediction analysis based on advanced statistics, 
    team performance metrics, and machine learning models. Our system uses an advanced ensemble model 
    combining XGBoost, LightGBM, and Random Forest classifiers for accurate game outcome predictions.
    """)
    
    # Sidebar for team selection
    st.sidebar.header("🏀 Team Selection")
    
    # Common NBA team names for easier selection
    nba_teams = [
        "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", 
        "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
        "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
        "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
        "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
        "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
        "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
        "Utah Jazz", "Washington Wizards"
    ]
    
    # Team selection inputs
    home_team = st.sidebar.selectbox(
        "🏠 Select Home Team:",
        options=[""] + nba_teams,
        index=0
    )
    
    away_team = st.sidebar.selectbox(
        "✈️ Select Away Team:",
        options=[""] + nba_teams,
        index=0
    )
    
    # Alternative text input for custom team names
    st.sidebar.markdown("---")
    st.sidebar.subheader("Or enter custom team names:")
    
    custom_home = st.sidebar.text_input("Home Team (custom):", placeholder="e.g., Portland Trail Blazers")
    custom_away = st.sidebar.text_input("Away Team (custom):", placeholder="e.g., Oklahoma City Thunder")
    
    # Use custom names if provided, otherwise use dropdown selections
    final_home_team = custom_home.strip() if custom_home.strip() else home_team
    final_away_team = custom_away.strip() if custom_away.strip() else away_team
    
    # Prediction button
    predict_button = st.sidebar.button("🔮 Generate Prediction", type="primary")
    
    # Main content area
    if predict_button:
        if not final_home_team or not final_away_team:
            st.error("❌ Please select or enter both teams before generating a prediction.")
            return
        
        if final_home_team == final_away_team:
            st.error("❌ Please select two different teams.")
            return
        
        # Show loading state
        with st.spinner(f"🔄 Analyzing matchup: {final_home_team} vs {final_away_team}..."):
            try:
                # Get prediction data from your existing predictor
                prediction_data = get_prediction_data_for_teams(final_home_team, final_away_team)

                # Check if predictor.py returned an error
                if 'error' in prediction_data:
                    st.error(f"❌ Prediction Error: {prediction_data['error']}")
                    return # Stop further processing
                
                if prediction_data is None: # This case might be less likely if predictor.py always returns a dict
                    st.error("❌ Could not find data for this matchup. Please ensure both team names are correct and that historical data exists for these teams.")
                    return
                
                # Generate detailed LLM summary
                llm_summary = generate_game_summary(prediction_data)
                
                if llm_summary is None:
                    st.error("❌ Failed to generate AI summary. Please check your API configuration.")
                    return
                
                # Display results
                display_prediction_results(prediction_data, llm_summary, final_home_team, final_away_team)
                
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {str(e)}")
                st.exception(e)
    
    else:
        # Show welcome content when no prediction is being made
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 🎯 How It Works
            
            1. **Select Teams**: Choose home and away teams from the dropdown or enter custom names
            2. **AI Analysis**: Our ML model analyzes historical data, team statistics, and performance metrics
            3. **Detailed Report**: Get a comprehensive 200+ word analysis with predictions and insights
            
            ### 📊 Features
            - **Advanced Analytics**: Team performance, head-to-head history, and key statistical factors
            - **AI-Powered Insights**: Natural language explanations of predictions and key factors
            """)

if __name__ == "__main__":
    main()
