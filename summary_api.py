from groq import Groq
import os
from dotenv import load_dotenv

"""
Generates a concise NBA game summary using Groq's LLM. Make sure to have the correct API key in the .env file.

Args:
    llm_prompt_data (dict): Dictionary with game prediction data.

Returns:
    str: Generated summary from LLM.
"""

# Load .env file to ensure OPENAI_API_KEY is available in the environment
load_dotenv()

# Use API_KEY for Groq, as specified by the user
api_key = os.getenv("API_KEY") 
client = None

if api_key:
    client = Groq(api_key=api_key) # Initialize Groq client
else:
    # This print will occur when summary_api.py is first imported/loaded by predictor.py
    # if the API key is not found. The function generate_game_summary will then also fail.
    print("Warning from summary_api.py: API_KEY (for Groq) not found in environment. Summary generation will likely fail.")

# Debug print (optional, can be removed after testing)
# print(f"DEBUG: In summary_api.py, loaded API_KEY: '{api_key[:5] if api_key else None}...'")

def generate_game_summary(llm_prompt_data):
    if 'error' in llm_prompt_data:
        error_message = llm_prompt_data['error']
        print(f"Error in summary_api.py: Cannot generate summary because prediction data contains an error: {error_message}")
        return f"Could not generate AI summary: {error_message}"

    if not client:
        print("Error in summary_api.py: Groq client not initialized. Missing API Key?")
        return None
        
    system_prompt = (
        "You are a professional NBA sports analyst generating detailed, insightful game predictions "
        "based on team statistics and machine learning models. Provide comprehensive analysis with "
        "at least 150 words that includes strategic insights, team strengths/weaknesses, and key factors."
    )

    user_prompt = f"""
Generate a detailed NBA game analysis based on the following prediction data. Your response must be at least 200 words and should provide comprehensive insights.

PREDICTION SUMMARY:
- Predicted Winner: {llm_prompt_data['predicted_winner_name']} 
- Win Probability: {llm_prompt_data['win_probability_percent']}%
- Matchup: {llm_prompt_data['home_team_name']} (Home) vs {llm_prompt_data['away_team_name']} (Away)

RECENT PERFORMANCE METRICS (Last ~10 games):

{llm_prompt_data['home_team_name']} (Home Team) Stats:
- Points per game: {llm_prompt_data['home_stats'].get('pts', 'N/A')}
- Field Goal Percentage: {llm_prompt_data['home_stats'].get('fg_pct', 'N/A')}
- 3-Point Percentage: {llm_prompt_data['home_stats'].get('fg3_pct', 'N/A')}
- Assists per game: {llm_prompt_data['home_stats'].get('ast', 'N/A')}
- Rebounds per game: {llm_prompt_data['home_stats'].get('reb', 'N/A')}
- Turnovers per game: {llm_prompt_data['home_stats'].get('tov', 'N/A')}
- Win Percentage: {llm_prompt_data['home_stats'].get('win_pct', 'N/A')}

{llm_prompt_data['away_team_name']} (Away Team) Stats:
- Points per game: {llm_prompt_data['away_stats'].get('pts', 'N/A')}
- Field Goal Percentage: {llm_prompt_data['away_stats'].get('fg_pct', 'N/A')}
- 3-Point Percentage: {llm_prompt_data['away_stats'].get('fg3_pct', 'N/A')}
- Assists per game: {llm_prompt_data['away_stats'].get('ast', 'N/A')}
- Rebounds per game: {llm_prompt_data['away_stats'].get('reb', 'N/A')}
- Turnovers per game: {llm_prompt_data['away_stats'].get('tov', 'N/A')}
- Win Percentage: {llm_prompt_data['away_stats'].get('win_pct', 'N/A')}

KEY PREDICTION FACTORS:
{llm_prompt_data['top_global_features']}

ANALYSIS REQUIREMENTS:
Please provide a comprehensive analysis that includes:

1. **Prediction Rationale**: Explain why the model predicts {llm_prompt_data['predicted_winner_name']} to win with {llm_prompt_data['win_probability_percent']}% confidence.

2. **Team Comparison**: Compare the key strengths and weaknesses of both teams based on their recent performance metrics. Highlight which statistical areas give the predicted winner their advantage.

3. **Key Factors Analysis**: Discuss how the most important statistical factors (listed above) influence this prediction. Explain what these metrics tell us about each team's playing style and effectiveness.

4. **Strategic Insights**: Provide insights into what each team needs to do to win, potential game flow scenarios, and areas where the underdog could potentially upset the prediction.

5. **Home Court Advantage**: Consider the impact of {llm_prompt_data['home_team_name']} playing at home and how this might affect the outcome.

Ensure your response is detailed, analytical, and provides valuable insights for basketball fans and analysts. Minimum 200 words required.
"""

    try:
        # Increased max_tokens to ensure at least 200 words
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500  # Increased from 250 to ensure at least 200 words
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating summary with Groq in summary_api.py: {e}")
        return None
