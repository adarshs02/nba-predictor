import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import os
from dotenv import load_dotenv
import joblib
import matplotlib.pyplot as plt
import numpy as np
from bird_framework import BIRDModel

"""
Predicts the outcome of an NBA game using a XGBoost Classifier. Current accuracy is right under 60%. There are no player based predictions yet. It only takes into account team stats.
"""

# Ensure summary_api import is correctly placed and attempted
try:
    from summary_api import generate_game_summary
    summary_api_available = True
    # ADDING DIAGNOSTIC PRINT HERE
    import summary_api # Import the module itself to check its path
    print(f"DEBUG: predictor.py - Successfully imported summary_api. Path: {summary_api.__file__}")
except ImportError as e:
    # MODIFIED DIAGNOSTIC PRINT HERE
    print(f"DEBUG: predictor.py - Failed to import summary_api. Error: {e}")
    summary_api_available = False
    summary_api = None # To avoid NameError if __file__ was attempted on a non-existent module object.

# Load environment variables from .env file (for OPENAI_API_KEY, DB_NAME, etc.)
load_dotenv()

# Initialize variables
DB_NAME = os.getenv("DB_NAME", "nba_data.db")
MODEL_FILENAME = "nba_predictor_model.joblib"
BIRD_MODEL_FILENAME = "nba_predictor_bird_model.joblib"
FEATURES_FILENAME = "feature_importances.joblib"
home_team_name = "Denver Nuggets"
away_team_name = "Oklahoma City Thunder"

def load_data():
    # ... (load_data function remains largely the same, ensuring teams_info_df is loaded) ...
    """Loads all necessary data from the SQLite database."""
    print(f"Loading data from {DB_NAME}...")
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        
        games_df = pd.read_sql_query("SELECT * FROM games", conn)
        print(f"Loaded {len(games_df)} entries from 'games' table.")
        if games_df.empty:
            print("Warning: 'games' table is empty. Ensure data_collector.py has run successfully.")

        player_logs_df = pd.read_sql_query("SELECT * FROM player_game_logs", conn)
        print(f"Loaded {len(player_logs_df)} entries from 'player_game_logs' table.")
        if player_logs_df.empty:
            print("Warning: 'player_game_logs' table is empty.")

        team_logs_df = pd.read_sql_query("SELECT * FROM team_game_logs", conn)
        print(f"Loaded {len(team_logs_df)} entries from 'team_game_logs' table.")
        if team_logs_df.empty:
            print("Warning: 'team_game_logs' table is empty.")

        teams_info_df = pd.read_sql_query("SELECT * FROM teams", conn)
        print(f"Loaded {len(teams_info_df)} entries from 'teams' table.")
        if teams_info_df.empty:
            print("Warning: 'teams' table is empty (needed for team names and IDs).")

        return {
            "games": games_df,
            "player_logs": player_logs_df,
            "team_logs": team_logs_df,
            "teams_info": teams_info_df
        }

    except sqlite3.OperationalError as e:
        print(f"Database error: {e}. Make sure '{DB_NAME}' exists and data_collector.py has run successfully.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None
    finally:
        if conn:
            conn.close()

def calculate_team_rolling_stats(team_id, game_date, team_logs_df, window_size=10):
    # ... (calculate_team_rolling_stats function remains the same) ...
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'])
    game_date = pd.to_datetime(game_date)

    past_games = team_logs_df[
        (team_logs_df['team_id'] == team_id) &
        (team_logs_df['game_date'] < game_date)
    ].sort_values(by='game_date', ascending=False).head(window_size)

    if past_games.empty:
        return pd.Series(dtype='float64') 

    stats_to_average = ['pts', 'fg_pct', 'fg3_pct', 'ast', 'reb', 'tov']
    
    for col in stats_to_average:
        if col in past_games.columns:
            past_games[col] = pd.to_numeric(past_games[col], errors='coerce')
        else: 
            past_games[col] = pd.NA 
    
    rolling_stats = past_games[stats_to_average].mean(numeric_only=True)
    
    # Calculate win_pct if 'wl' column exists
    if 'wl' in past_games.columns:
        rolling_stats['win_pct'] = (past_games['wl'] == 'W').mean()
    else:
        rolling_stats['win_pct'] = pd.NA # Or 0 or some other default

    # Prefix stats with 'avg_' to match expected feature names if needed by LLM prompt structure
    rolling_stats = rolling_stats.add_prefix('avg_')
    return rolling_stats

# Algorithm now improved with BIRD (Bayesian Inference for Reliable Decisions) framework
def feature_engineering(raw_data_dict):
    # ... (feature_engineering function updated to set game_id as index for X) ...
    if raw_data_dict is None:
        print("Raw data is missing, cannot perform feature engineering.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    games_df = raw_data_dict.get("games")
    team_logs_df = raw_data_dict.get("team_logs")
    teams_info_df = raw_data_dict.get("teams_info")

    if games_df is None or games_df.empty or \
       team_logs_df is None or team_logs_df.empty or \
       teams_info_df is None or teams_info_df.empty:
        print("Games, Team Logs, or Teams Info DataFrame is missing or empty. Cannot proceed.")
        # Fallback to return empty structures matching the expected output
        return pd.DataFrame(), pd.Series(), pd.DataFrame()


    print("Starting feature engineering...")

    games_df['game_date'] = pd.to_datetime(games_df['game_date'])
    games_df = games_df.sort_values(by=['game_date', 'game_id'])

    games_df['home_team_score'] = pd.to_numeric(games_df['home_team_score'], errors='coerce').fillna(0)
    games_df['away_team_score'] = pd.to_numeric(games_df['away_team_score'], errors='coerce').fillna(0)
    games_df['HOME_TEAM_WINS'] = (games_df['home_team_score'] > games_df['away_team_score']).astype(int)

    team_id_to_name = teams_info_df.set_index('id')['full_name'].to_dict()

    all_features_list = []
    window_size = 10
    team_logs_df['game_date'] = pd.to_datetime(team_logs_df['game_date'], errors='coerce')

    for index, game_row in games_df.iterrows():
        game_id = game_row['game_id']
        home_team_id = game_row['home_team_id']
        away_team_id = game_row['away_team_id']
        current_game_date = game_row['game_date']

        home_team_stats = calculate_team_rolling_stats(home_team_id, current_game_date, team_logs_df, window_size)
        away_team_stats = calculate_team_rolling_stats(away_team_id, current_game_date, team_logs_df, window_size)

        game_features = {
            'game_id': game_id, # Keep game_id for now, will be index later
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_name': team_id_to_name.get(home_team_id, 'Unknown'),
            'away_team_name': team_id_to_name.get(away_team_id, 'Unknown')
        }
        for stat_name, value in home_team_stats.items(): # stat_name will be avg_pts etc.
            game_features[f'home_{stat_name}'] = value 
        for stat_name, value in away_team_stats.items(): # stat_name will be avg_pts etc.
            game_features[f'away_{stat_name}'] = value
        
        all_features_list.append(game_features)

    features_df_with_details = pd.DataFrame(all_features_list)
    features_df_with_details = features_df_with_details.fillna(0) # Fill NaNs after stat calculation

    print(f"Feature engineering complete. Generated {len(features_df_with_details)} feature sets.")
    
    final_df = pd.merge(features_df_with_details, games_df[['game_id', 'HOME_TEAM_WINS']], on='game_id', how='inner')
    
    if final_df.empty:
        print("No data after merging features and target. Check game_id alignment.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    target_final = final_df['HOME_TEAM_WINS']
    
    # Set game_id as index for features_for_model (X)
    # This X will be used for training and for looking up specific games for prediction
    features_for_model = final_df.set_index('game_id').drop(columns=[
        'HOME_TEAM_WINS', 
        'home_team_id', 'away_team_id',
        'home_team_name', 'away_team_name' 
    ])

    full_game_details_df = final_df[['game_id', 'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name', 'HOME_TEAM_WINS']].set_index('game_id')

    return features_for_model, target_final, full_game_details_df


def train_model(X_train, y_train, calibration_X=None, calibration_y=None):
    """Trains the prediction model, saves it and feature importances, and returns them.
    Now enhanced with BIRD framework for Bayesian uncertainty quantification.
    
    Args:
        X_train: Training features
        y_train: Training labels
        calibration_X: Optional separate calibration set features
        calibration_y: Optional separate calibration set labels
    """
    # Initialize and train the model using XGBoost for improved accuracy
    print("Using XGBoost for improved prediction accuracy")
    base_model = XGBClassifier(
        n_estimators=300,       # Number of boosting rounds
        learning_rate=0.05,     # Slower learning rate for better generalization
        max_depth=6,            # Control tree depth to prevent overfitting
        min_child_weight=3,     # Minimum sum of instance weight needed in a child
        gamma=0.1,              # Minimum loss reduction for partition
        subsample=0.8,          # Use 80% of data per tree (prevents overfitting)
        colsample_bytree=0.8,   # Use 80% of features per tree (prevents overfitting)
        objective='binary:logistic',  # Binary classification with logistic function
        scale_pos_weight=1,     # Balance positive/negative weights
        random_state=42,        # Reproducibility
        use_label_encoder=False,# Avoid deprecation warning
        eval_metric='logloss'   # Evaluation metric
    )
    base_model.fit(X_train, y_train)
    
    # Save the base model for future use
    joblib.dump(base_model, MODEL_FILENAME)
    print(f"Base model saved to {MODEL_FILENAME}")
    
    # Save the feature importances
    feature_names = X_train.columns
    importances = base_model.feature_importances_
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    joblib.dump(feature_importances, FEATURES_FILENAME)
    print(f"Feature importances saved to {FEATURES_FILENAME}")
    
    # Create a BIRD model with the base model and Monte Carlo sampling for uncertainty
    # Use a low calibration_influence value (0.3) to improve accuracy while still benefiting from calibration
    bird_model = BIRDModel(base_model=base_model, calibration_influence=0.3, n_samples=30, temperature=1.0)
    
    # If we have separate calibration data, use it to calibrate the uncertainty estimates
    if calibration_X is not None and calibration_y is not None:
        print("Calibrating BIRD model with separate calibration data...")
        calibration_data = calibration_X.copy()
        calibration_data['actual_winner'] = calibration_y
        bird_model.calibrate(calibration_data)
    elif len(X_train) > 100:  # If enough training data, use a portion for calibration
        print("Creating calibration dataset with guaranteed mixed correctness...")
        
        # Make predictions on all training data
        train_predictions = base_model.predict(X_train)
        train_probs = base_model.predict_proba(X_train)[:, 1]  # Probability of class 1
        
        # Check where model was correct and incorrect (using numpy arrays for safety)
        predictions_array = np.array(train_predictions)
        actuals_array = np.array(y_train)
        is_correct = predictions_array == actuals_array
        
        # Find positions (not DataFrame indices) where model was correct and incorrect
        correct_positions = np.where(is_correct)[0]
        incorrect_positions = np.where(~is_correct)[0]
        
        print(f"\n---------- CALIBRATION DIAGNOSTICS ----------")
        print(f"Training set - Total: {len(y_train)}, Correct: {len(correct_positions)}, Incorrect: {len(incorrect_positions)}")
        print(f"Base model accuracy on FULL training set: {100 * len(correct_positions) / len(y_train):.2f}%")
        
        # Print distribution of actual classes in training data
        classes, counts = np.unique(actuals_array, return_counts=True)
        print(f"Actual class distribution in training data: {dict(zip(classes, counts))}")
        
        # Analyze prediction confidence distribution
        confidence_correct = [train_probs[i] if predictions_array[i] == 1 else 1-train_probs[i] for i in correct_positions]
        confidence_incorrect = [train_probs[i] if predictions_array[i] == 1 else 1-train_probs[i] for i in incorrect_positions]
        
        # Calculate statistics on confidence
        if len(confidence_correct) > 0:
            avg_confidence_correct = sum(confidence_correct) / len(confidence_correct)
            print(f"Average confidence when correct: {avg_confidence_correct:.4f}")
        
        if len(confidence_incorrect) > 0:
            avg_confidence_incorrect = sum(confidence_incorrect) / len(confidence_incorrect)
            print(f"Average confidence when incorrect: {avg_confidence_incorrect:.4f}")
            
        print("-----------------------------------------------\n")
        
        # Ensure we have both correct and incorrect examples
        if len(correct_positions) == 0 or len(incorrect_positions) == 0:
            print("WARNING: Model is either 100% accurate or 100% inaccurate on training data.")
            print("Cannot calibrate without examples of both correct and incorrect predictions.")
            print("Using raw probabilities without calibration.")
        else:
            # Calculate calibration set size (30% of training set, with a minimum size of 50)
            cal_size = max(min(int(len(X_train) * 0.3), 500), 50)
            print(f"Using a calibration set of size {cal_size}")
            
            # Determine sampling counts to ensure a mix of correct and incorrect predictions
            # Aim for at least 25% of the minority class (correct or incorrect predictions)
            minority_count = min(len(correct_positions), len(incorrect_positions))
            majority_count = max(len(correct_positions), len(incorrect_positions))
            
            # Calculate how many samples to take from each group
            minority_samples = max(int(cal_size * 0.25), 10)  # At least 10 samples from minority
            minority_samples = min(minority_samples, minority_count)  # But no more than available
            majority_samples = min(cal_size - minority_samples, majority_count)
            
            # Sample from correct and incorrect predictions
            if len(correct_positions) <= len(incorrect_positions):
                sampled_correct = np.random.choice(correct_positions, size=minority_samples, replace=False)
                sampled_incorrect = np.random.choice(incorrect_positions, size=majority_samples, replace=False)
            else:
                sampled_correct = np.random.choice(correct_positions, size=majority_samples, replace=False)
                sampled_incorrect = np.random.choice(incorrect_positions, size=minority_samples, replace=False)
            
            # Combine samples and shuffle
            calibration_indices = np.concatenate([sampled_correct, sampled_incorrect])
            np.random.shuffle(calibration_indices)
            
            # Create calibration dataset
            calibration_X = X_train.iloc[calibration_indices].copy()
            calibration_y = y_train.iloc[calibration_indices].copy()
            
            # --- Debug: Check base model accuracy on this calibration set ---
            print("\n---------- CALIBRATION SAMPLE ANALYSIS ----------")
            print(f"Sampled {minority_samples} from minority class and {majority_samples} from majority class")
            print(f"Calibration set size: {len(calibration_indices)} samples")
            
            # Calculate fraction of training data in calibration
            print(f"Calibration set is {100 * len(calibration_indices) / len(X_train):.1f}% of training data")
            
            # Check balance of correct/incorrect predictions in calibration set
            cal_correct = np.sum(is_correct[calibration_indices])
            cal_incorrect = len(calibration_indices) - cal_correct
            print(f"Calibration samples: {cal_correct} correct, {cal_incorrect} incorrect predictions")
            print(f"Correctness ratio: {cal_correct/len(calibration_indices):.2f} correct, {cal_incorrect/len(calibration_indices):.2f} incorrect")
            
            # Run predictions on calibration set
            cal_predictions = base_model.predict(calibration_X)
            cal_accuracy = accuracy_score(calibration_y, cal_predictions)
            print(f"Base model accuracy on calibration samples: {cal_accuracy*100:.2f}%")
            
            # Calculate class distributions
            y_cal_1d = calibration_y.values if isinstance(calibration_y, pd.Series) else calibration_y
            pred_cal_1d = cal_predictions
            
            actual_counts = np.bincount(y_cal_1d.astype(int))
            pred_counts = np.bincount(pred_cal_1d.astype(int))
            
            print(f"Actual outcomes in calibration set (0s, 1s): {actual_counts}")
            print(f"Predicted outcomes by base model (0s, 1s): {pred_counts}")
            print("-----------------------------------------------\n")
            
            # Verify calibration target diversity (most important check)
            is_correct = (y_cal_1d == pred_cal_1d)
            correct_count = np.sum(is_correct)
            incorrect_count = len(is_correct) - correct_count
            
            print(f"DEBUG: Calibration target distribution - Correct: {correct_count}, Incorrect: {incorrect_count}")
            print(f"DEBUG: This is what the LogisticRegression calibration model will train on")
            
            if correct_count == 0 or incorrect_count == 0:
                print("ERROR: Failed to create a calibration set with both correct and incorrect predictions.")
                print("Using raw probabilities without calibration.")
            else:
                # If we have a good mix, proceed with calibration
                bird_model.calibrate(X_cal=calibration_X, y_cal=calibration_y)
    
    # Save the BIRD model
    bird_model.save(BIRD_MODEL_FILENAME)
    print(f"BIRD model saved to {BIRD_MODEL_FILENAME}")
    
    return bird_model, feature_importances


def evaluate_model(model, X_test, y_test):
    """Evaluates model performance on test data with additional metrics for BIRD model."""
    if hasattr(model, 'base_model'):
        # It's a BIRD model
        print("Evaluating BIRD model...")
        base_model = model.base_model
        
        # Evaluate base model accuracy
        y_pred_base = base_model.predict(X_test)
        base_accuracy = accuracy_score(y_test, y_pred_base)
        print(f"\nBase Model Accuracy on Test Data: {base_accuracy * 100:.2f}%")
        
        # Evaluate BIRD model with uncertainty
        correct_predictions = 0
        total_predictions = len(X_test)
        reliability_scores = []
        entropies = []
        
        for i in range(total_predictions):
            features = X_test.iloc[[i]]
            true_label = y_test.iloc[i]
            
            # Get prediction with uncertainty
            result = model.predict_with_uncertainty(features)
            pred_label = result['prediction']
            reliability = result['reliability_score']
            entropy = result['entropy']
            
            reliability_scores.append(reliability)
            entropies.append(entropy)
            
            if pred_label == true_label:
                correct_predictions += 1
        
        bird_accuracy = correct_predictions / total_predictions
        print(f"BIRD Model Accuracy on Test Data: {bird_accuracy * 100:.2f}%")
        print(f"Average Reliability Score: {np.mean(reliability_scores):.4f}")
        print(f"Average Entropy: {np.mean(entropies):.4f}")
        
        # Plot reliability distribution
        plt.figure(figsize=(10, 6))
        plt.hist(reliability_scores, bins=20, alpha=0.7)
        plt.title('Distribution of Prediction Reliability Scores')
        plt.xlabel('Reliability Score')
        plt.ylabel('Count')
        plt.savefig('reliability_distribution.png')
        print("Saved reliability distribution plot to 'reliability_distribution.png'")
        
        return {
            'base_accuracy': base_accuracy,
            'bird_accuracy': bird_accuracy,
            'reliability_scores': reliability_scores,
            'entropies': entropies
        }
    else:
        # Regular model evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")
        return {'accuracy': accuracy}

def predict_game(model, game_features_df, home_team_name="Home Team", away_team_name="Away Team", top_n_features=5, feature_importances=None):
    """Predicts outcome, shows key factors, saves data for LLM, and calls summary_api to generate a game summary.
    Now enhanced with BIRD framework for Bayesian uncertainty quantification.
    """
    # Handle both single row DataFrame and Series
    if isinstance(game_features_df, pd.DataFrame) and len(game_features_df) == 1:
        game_features = game_features_df.iloc[0]
    elif isinstance(game_features_df, pd.Series):
        game_features = game_features_df
    else:
        print("Error: prediction requires either a single-row DataFrame or a Series of features.")
        return None

    # --- Align input features with the model's expected feature names ---
    actual_sklearn_model = model.base_model if hasattr(model, 'base_model') else model

    if not hasattr(actual_sklearn_model, 'feature_names_in_') or actual_sklearn_model.feature_names_in_ is None:
        print("Warning: Model does not have 'feature_names_in_'. Using features as-is. This might lead to errors if features mismatch.")
        features_df_for_prediction = pd.DataFrame([game_features.values], columns=game_features.index)
    else:
        expected_feature_names = actual_sklearn_model.feature_names_in_
        aligned_game_features_series = game_features.reindex(expected_feature_names)
        
        missing_features = aligned_game_features_series[aligned_game_features_series.isna()].index.tolist()
        if missing_features:
            print(f"Warning: {len(missing_features)} expected features were missing in input and filled with 0.0 for prediction: {missing_features}")
            aligned_game_features_series = aligned_game_features_series.fillna(0.0)
            
        extra_features = list(set(game_features.index) - set(expected_feature_names))
        if extra_features:
            print(f"Warning: {len(extra_features)} input features were not expected by the model and were ignored: {extra_features}")

        features_df_for_prediction = pd.DataFrame([aligned_game_features_series.values], columns=expected_feature_names)
    # --- End of alignment logic ---

    class_names = [away_team_name, home_team_name]  # 0=away wins, 1=home wins

    if hasattr(model, 'predict_with_uncertainty'): # BIRD model
        print("Using BIRD framework for prediction with uncertainty quantification...")
        result = model.predict_with_uncertainty(features_df_for_prediction)
        
        prediction = result['prediction']
        calibrated_probs = result['calibrated_probabilities']
        # raw_probs = result['raw_probabilities'] # Not directly used later, can be kept if needed for debugging
        reliability_score = result['reliability_score']
        entropy = result['entropy']
        
        home_win_prob = calibrated_probs[1]
        away_win_prob = calibrated_probs[0]
        
        fig = model.visualize_uncertainty(result, class_names=class_names, 
                                         title=f"{away_team_name} vs {home_team_name} Prediction")
        fig.savefig('prediction_uncertainty.png')
        print("Saved uncertainty visualization to 'prediction_uncertainty.png'")
    else: # Legacy model
        print("Using regular model for prediction (no uncertainty quantification)...")
        prediction = model.predict(features_df_for_prediction)[0]
        try:
            probs = model.predict_proba(features_df_for_prediction)[0]
            home_win_prob = probs[1]
            away_win_prob = probs[0]
            reliability_score = None
            entropy = None
        except Exception as e:
            print(f"Warning: Model doesn't support probability estimation or failed: {e}. Using decision only.")
            home_win_prob = 1.0 if prediction == 1 else 0.0
            away_win_prob = 1.0 if prediction == 0 else 0.0
            reliability_score = None
            entropy = None
    
    # Determine outcome
    if prediction == 1:
        predicted_winner_name_val = home_team_name
        predicted_loser_name_val = away_team_name
        win_probability_val = home_win_prob
    else:
        predicted_winner_name_val = away_team_name
        predicted_loser_name_val = home_team_name
        win_probability_val = away_win_prob
        
    print("\n" + "=" * 50)
    print(f"PREDICTION: {predicted_winner_name_val} will {'WIN' if win_probability_val >= 0.8 else 'LIKELY WIN'} against {predicted_loser_name_val}")
    print(f"Win Probability: {win_probability_val:.1%}")
    
    # If using BIRD, show uncertainty metrics
    if reliability_score is not None:
        confidence_level = ""
        if reliability_score > 0.8:
            confidence_level = "HIGH"
        elif reliability_score > 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
            
        print(f"Reliability Score: {reliability_score:.2f} ({confidence_level} CONFIDENCE)")
        print(f"Prediction Entropy: {entropy:.4f} (lower = more certain)")
    
    # Display key factors (if we have feature importances)
    if feature_importances is not None:
        print("\nKEY FACTORS:")
        
        # Handle both Series and DataFrame format
        if isinstance(feature_importances, pd.Series):
            top_features = feature_importances.index[:top_n_features].tolist()
        elif isinstance(feature_importances, pd.DataFrame) and 'feature' in feature_importances.columns:
            top_features = feature_importances.head(top_n_features)['feature'].tolist()
        else:
            top_features = []
            print("Warning: Feature importances format not recognized")
            
        # For each important feature
        for feature in top_features:
            if feature in game_features.index:
                value = game_features[feature]
                
                # Handle categorical or special features
                if "vs" in feature.lower() or "_vs_" in feature.lower():
                    teams = feature.split("_vs_") if "_vs_" in feature else feature.split(" vs ")
                    if len(teams) == 2:
                        team1, team2 = teams
                        print(f"- Matchup history between {team1} and {team2} is significant")
                    else:
                        print(f"- {feature}: {value}")
                elif "diff" in feature.lower() or "advantage" in feature.lower():
                    print(f"- {feature.replace('_', ' ').title()}: {value:.2f}")
                else:
                    # Attempt to make a readable statement about the feature
                    readable_feature = feature.replace('_', ' ').title()
                    if isinstance(value, (int, float)):
                        print(f"- {readable_feature}: {value:.2f}")
                    else:
                        print(f"- {readable_feature}: {value}")
    
    # Prepare stats for summary (extracting from game_features which is a Series)
    # Assuming game_features contains keys like 'home_avg_pts', 'away_avg_fg_pct' etc.
    home_stats_dict = {col.replace('home_avg_', ''): game_features[col] for col in game_features.index if col.startswith('home_avg_')}
    away_stats_dict = {col.replace('away_avg_', ''): game_features[col] for col in game_features.index if col.startswith('away_avg_')}

    # Save prediction data for LLM summary
    prediction_data = {
        'home_team_name': home_team_name,
        'away_team_name': away_team_name,
        'predicted_winner_name': predicted_winner_name_val, # Key changed for summary_api
        'predicted_loser_name': predicted_loser_name_val, # Added for consistency
        'win_probability_percent': f"{win_probability_val:.1%}", # Formatted for summary_api
        'win_probability_float': win_probability_val, # Keep original float for other uses
        'reliability_score': reliability_score,
        'entropy': entropy,
        'top_features': top_features if 'top_features' in locals() else [],
        'feature_values': {f: game_features.get(f, 'N/A') for f in (top_features if 'top_features' in locals() else [])},
        'home_stats': home_stats_dict, # Added for summary_api
        'away_stats': away_stats_dict  # Added for summary_api
    }
    
    try:
        with open('latest_prediction.json', 'w') as f:
            json.dump(prediction_data, f, indent=4)
        print("\nPrediction data saved to 'latest_prediction.json'")
    except Exception as e:
        print(f"Error saving prediction data: {e}")

    # Use summary_api to generate a text summary if available
    if 'summary_api_available' in globals() and summary_api_available:
        try:
            print("\nGenerating game summary...")
            summary = generate_game_summary(prediction_data)
            print("\nGAME SUMMARY:")
            print(summary)
        except Exception as e:
            print(f"Error generating summary: {e}")
    else:
        print("\nSummary API not available. Install summary_api for game summaries.")
    
    return prediction_data

# New helper function for team ID lookup
def get_team_info_by_name(team_name_query, teams_info_df):
    """Looks up team information by full name. Case-insensitive partial match first, then exact."""
    if teams_info_df is None or teams_info_df.empty:
        print("Teams info DataFrame is not available for lookup.")
        return None

    # Try case-insensitive exact match first
    match = teams_info_df[teams_info_df['full_name'].str.lower() == team_name_query.lower()]
    if not match.empty:
        return match.iloc[0] # Return the first match as a Series

    # Try case-insensitive partial match if no exact found
    match = teams_info_df[teams_info_df['full_name'].str.contains(team_name_query, case=False, na=False)]
    if not match.empty:
        print(f"Note: Partial match found for '{team_name_query}'. Using '{match.iloc[0]['full_name']}'.")
        return match.iloc[0]
        
    print(f"Warning: Team '{team_name_query}' not found in teams_info_df.")
    return None

if __name__ == "__main__":
    print("NBA Game Predictor with BIRD Framework")
    
    # Initialize this variable early to avoid NameError
    using_bird = False
    
    data = load_data()
    
    if not data:
        print("Data loading failed. Exiting.")
        exit(1)
    
    # More robust check for essential dataframes
    if not all(data.get(key) is not None and not data.get(key).empty for key in ["games", "team_logs", "teams_info"]):
        exit("Essential dataframes (games, team_logs, teams_info) are missing or empty. Exiting.")

    X, y, full_game_details_df = feature_engineering(data) 
    if X.empty or y.empty:
        exit("Feature engineering resulted in empty data. Cannot proceed. Exiting.")

    model = None
    feature_importances = None
    force_retrain = True # Set to True to retrain model with current features

    # Try to load the model and feature importances
    if os.path.exists(MODEL_FILENAME) and os.path.exists(FEATURES_FILENAME) and not force_retrain:
        print(f"Loading saved model from {MODEL_FILENAME}...")
        try:
            model = joblib.load(MODEL_FILENAME)
            print(f"Loading saved feature importances from {FEATURES_FILENAME}...")
            feature_importances = joblib.load(FEATURES_FILENAME)
            print("Model and feature importances loaded successfully.")
        except Exception as e:
            print(f"Error loading saved model or features: {e}. Will retrain.")
            model = None # Ensure model is None if loading failed
            feature_importances = None

    if model is None or feature_importances is None: 
        if len(X) < 2 or len(y) < 2: # Ensure enough data for splitting and training
            # Correctly indented exit call
            exit(f"Not enough data to train model. Found {len(X)} samples. Needs at least 2. Exiting.")
        test_size = 0.2
        
        # Handle stratification 
        can_stratify = True
        vc = y.value_counts() 
        
        if len(vc) > 1: # More than one class present
            if vc.min() < 5: # Very small count for one class
                can_stratify = False 
                print("Warning: One class has very few examples, stratification disabled.")
            else:
                print(f"Stratifying split by target values (counts: {vc.to_dict()})") 
        else: # Single class data
            can_stratify = False
            print("Warning: Only one class present in the target variable. Stratification is not applicable.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=42, 
                                                        stratify=y if can_stratify else None)
        
        if X_train.empty: # Primary check for train set
             exit("Training set is empty after split. This can happen with very small datasets. Exiting.")

        print("Training a new model with BIRD framework...")
        model, feature_importances = train_model(X_train, y_train)
        using_bird = True
        print("\nTop Global Feature Importances (from new model):")
        if isinstance(feature_importances, pd.DataFrame) and 'feature' in feature_importances.columns:
            print(feature_importances[['feature', 'importance']].head(10))
        else:
            print(feature_importances.head(10))

        if not X_test.empty and not y_test.empty:
            evaluate_model(model, X_test, y_test)
        else:
            print("\nSkipping model evaluation as the test set (or y_test) is empty.")
    else:
        print("\nTop Global Feature Importances:")
        if isinstance(feature_importances, pd.DataFrame) and 'feature' in feature_importances.columns:
            print(feature_importances[['feature', 'importance']].head(10))
        else:
            print(feature_importances.head(10))

    # --- Prediction by Team Names ---
    if model and not full_game_details_df.empty and data.get("teams_info") is not None:
        home_team_name_input = home_team_name
        away_team_name_input = away_team_name
        
        print(f"\nAttempting to find a past game for {home_team_name_input} vs {away_team_name_input} for prediction...")

        # Find the first game that matches the home and away team names
        # Ensure case-insensitivity if team names in DB might vary from input
        target_game_df = full_game_details_df[
            (full_game_details_df['home_team_name'].str.lower() == home_team_name_input.lower()) &
            (full_game_details_df['away_team_name'].str.lower() == away_team_name_input.lower())
        ]

        if not target_game_df.empty:
            game_id_to_predict = target_game_df.index[0] # game_id is the index
            print(f"Found game_id: {game_id_to_predict} for the matchup.")

            # Get features for this specific game
            # X has game_id as index
            if game_id_to_predict in X.index:
                single_game_features = X.loc[game_id_to_predict]
                
                # Get actual team names from full_game_details_df for consistent casing
                actual_home_name = target_game_df.loc[game_id_to_predict, 'home_team_name']
                actual_away_name = target_game_df.loc[game_id_to_predict, 'away_team_name']

                if using_bird:
                    print("Using BIRD framework for prediction with uncertainty quantification...")
                else:
                    print("Using regular model for prediction (no uncertainty quantification)...")
                    
                prediction_result = predict_game(model, single_game_features, 
                                  home_team_name=actual_home_name, 
                                  away_team_name=actual_away_name, 
                                  top_n_features=5, 
                                  feature_importances=feature_importances)
            else:
                print(f"Could not find features for game_id {game_id_to_predict} in the feature set (X).")
        else:
            print(f"No past game found in the database for {home_team_name_input} (Home) vs {away_team_name_input} (Away).")
            print("Try different team names or ensure data_collector.py has run for games involving these teams.")
    elif not model:
        print("\nModel not trained or loaded. Cannot make predictions by team names.")
    else:
        print("\nCannot make predictions by team names due to missing data (full_game_details_df or teams_info).")

    print("\nPredictor script completed.") 