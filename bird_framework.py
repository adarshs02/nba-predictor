"""
BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models

This module implements the BIRD framework for improving the reliability and
trustworthiness of LLM-based predictions by using Bayesian inference to 
quantify uncertainty.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import List, Dict, Tuple, Union, Optional
import joblib
import os
import time
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

class BIRDModel:
    """
    BIRD (Bayesian Inference for Reliable Decisions) framework for LLMs.
    
    This class provides methods for:
    1. Calibrating model confidence
    2. Quantifying prediction uncertainty
    3. Providing reliability scores for predictions
    4. Visualizing uncertainty
    """
    
    def __init__(self, base_model=None, n_samples: int = 30, temperature: float = 1.0,
                 calibration_data: Optional[pd.DataFrame] = None, calibration_influence: float = 0.5):
        """
        Initialize the BIRD framework.
        
        Args:
            base_model: The base ML model (e.g., RandomForest) to wrap with BIRD
            n_samples: Number of samples for Monte Carlo estimation of uncertainty
            temperature: Temperature parameter to control diversity of sampling
            calibration_data: Optional data to calibrate the uncertainty estimates
            calibration_influence: How much influence calibration has on final predictions (0-1)
                                  0 = use only raw probabilities, 1 = use only calibrated probabilities
        """
        self.base_model = base_model
        self.n_samples = n_samples
        self.temperature = temperature
        self.calibration_coef = 1.0  # Default no calibration
        self.calibration_intercept = 0.0
        self.calibration_model = None
        self.calibration_influence = calibration_influence
        self.use_hybrid_predictions = True  # New flag to use hybrid prediction approach
        
        if calibration_data is not None:
            self.calibrate(calibration_data)
    
    def calibrate(self, X_cal: pd.DataFrame, y_cal=None, true_labels_col: str = 'actual_winner') -> None:
        """
        Calibrate the uncertainty estimates using historical data.
        
        Args:
            X_cal: DataFrame with features for calibration
            y_cal: Series or array with true labels (optional)
            true_labels_col: Name of column containing true outcomes if y_cal is None
        """
        if self.base_model is None:
            raise ValueError("Base model must be set before calibration")
            
        # Handle different ways calibration data might be passed
        if y_cal is None and true_labels_col in X_cal.columns:
            # Extract y from the input dataframe
            y_cal = X_cal[true_labels_col]
            X_cal = X_cal.drop(columns=[true_labels_col])
        elif y_cal is None:
            raise ValueError(f"No target labels provided for calibration. Either provide y_cal or include '{true_labels_col}' column in X_cal.")
            
        # Ensure y_cal is in the right format (np.array or Series, not DataFrame)
        if isinstance(y_cal, pd.DataFrame):
            # If y_cal is a DataFrame, convert to Series
            if y_cal.shape[1] == 1:
                y_cal = y_cal.iloc[:, 0]
            else:
                raise ValueError("y_cal should be a Series or 1D array, not a multi-column DataFrame")
        
        print(f"Starting calibration with {len(X_cal)} samples...")
        
        # Get uncalibrated probabilities
        raw_probs = self.base_model.predict_proba(X_cal)
        
        # Create the calibration target: 1 if prediction was correct, 0 if incorrect
        predictions = np.argmax(raw_probs, axis=1)
        
        # Ensure y_cal is in numpy format for comparison with predictions
        if isinstance(y_cal, pd.Series):
            y_cal_values = y_cal.values
        else:
            y_cal_values = np.array(y_cal)
            
        # Handle potential type mismatches by converting both to same type
        # First check if types are different
        if y_cal_values.dtype != predictions.dtype:
            # Convert both to int type to be safe
            y_cal_int = y_cal_values.astype(int)
            predictions_int = predictions.astype(int)
            is_correct = (y_cal_int == predictions_int)
        else:
            # If types already match, no need to convert
            is_correct = (y_cal_values == predictions)
        
        # Check class balance in calibration targets
        correct_count = np.sum(is_correct)
        incorrect_count = len(is_correct) - correct_count
        
        print(f"Calibration targets: {correct_count} correct, {incorrect_count} incorrect predictions")
        print(f"Correctness ratio: {100*correct_count/len(is_correct):.1f}% correct, {100*incorrect_count/len(is_correct):.1f}% incorrect")
        
        # Check if there are at least two classes for the calibration target
        if correct_count == 0 or incorrect_count == 0:
            print("Warning: Calibration data resulted in a single class for the target.")
            print("Will use Isotonic Regression with synthetic data augmentation.")
            
            # Create synthetic data points to allow calibration
            from sklearn.isotonic import IsotonicRegression
            
            # Extract confidence scores (probability of the predicted class)
            confidence = np.array([raw_probs[i, predictions[i]] for i in range(len(predictions))])
            
            # Create synthetic points with 0% and 100% confidence
            # to anchor the isotonic regression
            synthetic_confidence = np.concatenate([confidence, np.array([0.1, 0.9])])
            synthetic_correctness = np.concatenate([is_correct, np.array([0, 1])])
            
            # Use isotonic regression which is more robust for small/imbalanced datasets
            self.calibration_model = IsotonicRegression(out_of_bounds='clip')
            self.calibration_model.fit(synthetic_confidence, synthetic_correctness)
            print("Created isotonic regression calibration model with synthetic data augmentation.")
            
        else:
            # Use standard Platt scaling with logistic regression
            from sklearn.linear_model import LogisticRegression
            
            # Extract confidence scores (probability of the predicted class)
            confidence = np.array([raw_probs[i, predictions[i]] for i in range(len(predictions))])
            
            # Reshape for sklearn API
            confidence_reshaped = confidence.reshape(-1, 1)
            
            # Fit logistic regression for Platt scaling
            self.calibration_model = LogisticRegression(class_weight='balanced')
            self.calibration_model.fit(confidence_reshaped, is_correct)
            print("Created logistic regression calibration model.")
            
        # Test the calibration on the calibration set
        self._test_calibration(confidence, is_correct)
    
    def _test_calibration(self, confidence, is_correct):
        """
        Test the calibration on the given calibration set and report metrics.
        
        Args:
            confidence: Array of confidence scores (raw probabilities)
            is_correct: Array of binary values (1=correct, 0=incorrect)
        """
        if self.calibration_model is None:
            print("No calibration model available to test.")
            return
            
        try:
            # For logistic regression, we need to reshape
            if hasattr(self.calibration_model, 'predict_proba'):
                confidence_reshaped = confidence.reshape(-1, 1)
                calibrated_probs = self.calibration_model.predict_proba(confidence_reshaped)[:, 1]
            else:  # For isotonic regression
                calibrated_probs = self.calibration_model.predict(confidence)
                
            # Calculate average confidence before and after calibration
            avg_raw_confidence = np.mean(confidence)
            avg_calibrated_confidence = np.mean(calibrated_probs)
            
            # Calculate accuracy
            accuracy = np.mean(is_correct)
            
            print(f"Calibration test results:")
            print(f"  - Accuracy: {accuracy:.4f}")
            print(f"  - Avg raw confidence: {avg_raw_confidence:.4f}")
            print(f"  - Avg calibrated confidence: {avg_calibrated_confidence:.4f}")
            
            # Check if calibration improved the alignment between confidence and accuracy
            raw_diff = abs(avg_raw_confidence - accuracy)
            cal_diff = abs(avg_calibrated_confidence - accuracy)
            
            if cal_diff < raw_diff:
                print(f"  - Calibration IMPROVED confidence-accuracy alignment by {raw_diff-cal_diff:.4f}")
            else:
                print(f"  - Calibration WORSENED confidence-accuracy alignment by {cal_diff-raw_diff:.4f}")
                
        except Exception as e:
            print(f"Error testing calibration: {e}")
    
    def predict_with_uncertainty(self, features: pd.DataFrame) -> Dict:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            features: DataFrame of features for prediction
            
        Returns:
            Dictionary containing predictions, probabilities, and uncertainty metrics
        """
        if self.base_model is None:
            raise ValueError("Base model not initialized")
        
        # Generate predictions with dropout-based sampling for uncertainty
        probs_samples = []
        
        # For sklearn models, we'll simulate sampling by bootstrapping features
        for _ in range(self.n_samples):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(
                range(len(features)), size=len(features), replace=True
            )
            bootstrap_features = features.iloc[bootstrap_indices].reset_index(drop=True)
            
            # Get probabilities
            sample_probs = self.base_model.predict_proba(bootstrap_features)
            probs_samples.append(sample_probs[0])  # First row for single prediction
        
        # Convert to numpy array
        probs_array = np.array(probs_samples)
        
        # Calculate mean probabilities
        mean_probs = np.mean(probs_array, axis=0)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            logits = np.log(mean_probs + 1e-10)
            scaled_logits = logits / self.temperature
            mean_probs = softmax(scaled_logits)
        
        # Calculate uncertainty metrics
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        
        # Calculate variation in predictions
        prediction_variance = np.var(np.argmax(probs_array, axis=1))
        
        # Apply calibration if a model exists
        if self.calibration_model:
            calibrated_probs = self._calibrate_probs(mean_probs)
            
            # Hybrid prediction approach
            if self.use_hybrid_predictions:
                # Blend raw and calibrated probabilities based on calibration_influence
                hybrid_probs = (1 - self.calibration_influence) * mean_probs + self.calibration_influence * calibrated_probs
                
                # Use the hybrid probabilities for prediction
                final_probs = hybrid_probs
                
                # Check if raw and calibrated models disagree on the prediction
                raw_pred = np.argmax(mean_probs)
                cal_pred = np.argmax(calibrated_probs)
                
                # If calibration would flip the prediction and raw prob is high enough, trust raw more
                if raw_pred != cal_pred and mean_probs[raw_pred] > 0.6:
                    # Reduce calibration influence for this prediction
                    adjusted_influence = self.calibration_influence * 0.5
                    final_probs = (1 - adjusted_influence) * mean_probs + adjusted_influence * calibrated_probs
            else:
                # Traditional approach: use calibrated probabilities directly
                final_probs = calibrated_probs
        else:
            print("Info: No calibration model available. Using raw probabilities.")
            calibrated_probs = mean_probs
            final_probs = mean_probs
        
        # Get predicted class
        predicted_class = np.argmax(final_probs)
        
        # Reliability score (higher = more reliable)
        reliability_score = 1.0 - (entropy / np.log(len(mean_probs)))
        
        return {
            'prediction': predicted_class,
            'raw_probabilities': mean_probs,
            'calibrated_probabilities': calibrated_probs,
            'final_probabilities': final_probs,  # New field with hybrid probabilities
            'entropy': entropy,
            'prediction_variance': prediction_variance,
            'reliability_score': reliability_score,
            'sample_predictions': probs_array
        }
        
    def _calibrate_probs(self, raw_class_probabilities: np.ndarray) -> np.ndarray:
        """Apply learned calibration model to raw class probabilities."""
        if self.calibration_model is None:
            return raw_class_probabilities

        # The calibration model was trained on: X=max_probability_of_a_prediction (confidence),
        # y=(actual_outcome == predicted_outcome_by_base_model).
        # It gives P(base_model_was_correct | confidence_of_base_model).

        # We have raw_class_probabilities for the current prediction, e.g., [0.3, 0.7]
        # Step 1: Get the confidence of the base model's prediction from these raw_class_probabilities
        confidence_current_prediction = np.max(raw_class_probabilities)
        
        # Step 2: Use the calibration model to get P(base_model_is_correct | confidence_current_prediction)
        if hasattr(self.calibration_model, 'predict_proba'):
            # For LogisticRegression, predict_proba returns [[P(incorrect), P(correct)]]
            prob_base_model_is_correct = self.calibration_model.predict_proba(np.array([[confidence_current_prediction]]))[0, 1]
        else:
            # For IsotonicRegression, predict directly returns the calibrated probability
            prob_base_model_is_correct = self.calibration_model.predict(np.array([confidence_current_prediction]))[0]

        # Step 3: How to adjust raw_class_probabilities using prob_base_model_is_correct?
        # This is the tricky part. Platt scaling typically adjusts the probability of the *predicted class*.
        # If predicted class C had raw prob P(C), its calibrated prob P_cal(C) is often derived.
        # One simple (but not always best) way: if base model predicts class k with P_raw(k),
        # and P(correct|confidence) = P_cal_correct, then scale P_raw(k) towards P_cal_correct.
        # A more standard way is to adjust the logits.

        # For simplicity and to avoid complex redistribution, many implementations of Platt scaling
        # directly calibrate the probability of the class that was predicted by the uncalibrated model.
        # Let's assume the calibration model directly gives the calibrated probability for the *predicted class*.
        # This is an approximation. A full recalibration of the probability distribution is more complex.

        # If the calibration model outputs P(correct | confidence), and the model predicted class 'j' with confidence C_j,
        # then the calibrated probability of class 'j' could be P(correct | C_j).
        # The probabilities of other classes need to be adjusted to sum to 1.

        # Given the current setup, the LogisticRegression model gives P(correctness).
        # Let's use it to scale the original winning probability.
        predicted_class_index = np.argmax(raw_class_probabilities)
        original_predicted_prob = raw_class_probabilities[predicted_class_index]

        # More conservative calibration approach to avoid reducing accuracy
        # Only scale the probability moderately based on calibration model
        # If original_predicted_prob is high (> 0.7), apply less scaling
        if original_predicted_prob > 0.7:
            # Apply minimal scaling for high-confidence predictions
            adjustment_factor = min(0.8, prob_base_model_is_correct)
            # Don't let calibration reduce the probability too much
            calibrated_predicted_prob = max(original_predicted_prob * 0.9, 
                                          original_predicted_prob * adjustment_factor)
        else:
            # For lower confidence predictions, apply more calibration
            calibrated_predicted_prob = original_predicted_prob * prob_base_model_is_correct
        
        # Adjust other probabilities proportionally to maintain sum-to-1
        # This is a common simplification but can be problematic.
        calibrated_probabilities = np.copy(raw_class_probabilities)
        if predicted_class_index < len(calibrated_probabilities): # Check index bounds
            # Calculate the sum of probabilities of other classes
            sum_other_probs = np.sum(raw_class_probabilities) - original_predicted_prob
            
            # If sum_other_probs is zero (e.g., 100% confidence in one class), avoid division by zero
            if sum_other_probs < 1e-9: # Effectively zero
                if original_predicted_prob > (1.0 - 1e-9): # If one class was 100%
                     calibrated_probabilities[predicted_class_index] = prob_base_model_is_correct # Calibrated prob for this class
                     # Other classes remain 0, but need to ensure sum to 1 if prob_base_model_is_correct is not 1
                     # This simple scaling might not preserve sum-to-1 if prob_base_model_is_correct is not 1.
                     # A better approach is needed if this path is taken. 
                     # For now, if original was 100%, let calibrated be P(correct), and others 1-P(correct) / (N-1)
                     if len(calibrated_probabilities) > 1:
                        other_prob_val = (1.0 - prob_base_model_is_correct) / (len(calibrated_probabilities) -1)
                        for i in range(len(calibrated_probabilities)):
                            if i == predicted_class_index:
                                calibrated_probabilities[i] = prob_base_model_is_correct
                            else:
                                calibrated_probabilities[i] = other_prob_val
                     else: # Single class case
                        calibrated_probabilities[predicted_class_index] = prob_base_model_is_correct # or 1.0

                # If original_predicted_prob was not 1.0 but sum_other_probs is zero (should not happen if N_class > 1)
                # Fallback to raw probabilities in this edge case.
                else: 
                    return raw_class_probabilities

            else:
                # Scale factor for other probabilities
                scale_factor_others = (1.0 - calibrated_predicted_prob) / sum_other_probs
                for i in range(len(calibrated_probabilities)):
                    if i == predicted_class_index:
                        calibrated_probabilities[i] = calibrated_predicted_prob
                    else:
                        calibrated_probabilities[i] = raw_class_probabilities[i] * scale_factor_others
            
            # Ensure probabilities sum to 1 (due to potential floating point issues)
            calibrated_probabilities /= np.sum(calibrated_probabilities)
            return calibrated_probabilities
        else:
            # Should not happen if raw_class_probabilities is valid
            return raw_class_probabilities
    
    def visualize_uncertainty(self, prediction_result: Dict, class_names: List[str],
                             title: str = "Prediction Uncertainty") -> plt.Figure:
        """
        Visualize the prediction uncertainty.
        
        Args:
            prediction_result: Output from predict_with_uncertainty
            class_names: List of class names (e.g., team names)
            title: Title for the visualization
            
        Returns:
            Matplotlib figure
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot probability distribution
        probs = prediction_result['calibrated_probabilities']
        ax1.bar(class_names, probs)
        ax1.set_ylabel('Probability')
        ax1.set_title('Prediction Probabilities')
        for i, v in enumerate(probs):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # Plot sample distributions 
        samples = prediction_result['sample_predictions']
        sns.violinplot(data=samples, ax=ax2)
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Uncertainty Distribution')
        ax2.set_xticklabels(class_names)
        
        reliability = prediction_result['reliability_score'] * 100
        fig.suptitle(f"{title}\nReliability Score: {reliability:.1f}%", fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def save(self, filepath: str) -> None:
        """Save the BIRD model"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BIRDModel':
        """Load a saved BIRD model"""
        return joblib.load(filepath)
