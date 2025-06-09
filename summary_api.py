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
    if not client:
        print("Error in summary_api.py: Groq client not initialized. Missing API Key?")
        return None
        
    system_prompt = (
        "You are a sports analyst generating concise, insightful NBA game summaries "
        "based on team statistics and win predictions. Use clear and professional language."
    )

    user_prompt = f"""
Prediction: {llm_prompt_data['predicted_winner_name']} to win with {llm_prompt_data['win_probability_percent']}% confidence.

Recent Performance Metrics (last ~10 games):

{llm_prompt_data['home_team_name']} Stats:
- Points per game: {llm_prompt_data['home_stats'].get('pts', 'N/A')}
- Field Goal %: {llm_prompt_data['home_stats'].get('fg_pct', 'N/A')}
- 3-Point %: {llm_prompt_data['home_stats'].get('fg3_pct', 'N/A')}
- Assists per game: {llm_prompt_data['home_stats'].get('ast', 'N/A')}
- Rebounds per game: {llm_prompt_data['home_stats'].get('reb', 'N/A')}
- Turnovers per game: {llm_prompt_data['home_stats'].get('tov', 'N/A')}
- Win Percentage: {llm_prompt_data['home_stats'].get('win_pct', 'N/A')}

{llm_prompt_data['away_team_name']} Stats:
- Points per game: {llm_prompt_data['away_stats'].get('pts', 'N/A')}
- Field Goal %: {llm_prompt_data['away_stats'].get('fg_pct', 'N/A')}
- 3-Point %: {llm_prompt_data['away_stats'].get('fg3_pct', 'N/A')}
- Assists per game: {llm_prompt_data['away_stats'].get('ast', 'N/A')}
- Rebounds per game: {llm_prompt_data['away_stats'].get('reb', 'N/A')}
- Turnovers per game: {llm_prompt_data['away_stats'].get('tov', 'N/A')}
- Win Percentage: {llm_prompt_data['away_stats'].get('win_pct', 'N/A')}

Top influential factors for this prediction model globally:
{llm_prompt_data['top_features']}

Based on the prediction and these statistics, analyze the {llm_prompt_data['predicted_winner_name']}'s chances.
Briefly mention their key strengths relative to the {llm_prompt_data['predicted_loser_name']}'s potential weaknesses,
or how the {llm_prompt_data['predicted_loser_name']} might challenge this prediction.
Keep the entire summary to a concise paragraph.
"""

    try:
        # Assuming Groq API for chat completions is similar to OpenAI's v1.x
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating summary with Groq in summary_api.py: {e}")
        return None
