"""
Churn Risk Classification Module
LightGBM-based binary classifier for customer churn prediction

Author: CLV Analytics Team
Date: February 2026

Methodology:
- Churn definition: No purchase in 90+ days (customizable)
- Model: LightGBM with class imbalance handling
- Output: Churn probability (0-1) per customer
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json
import dill

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ChurnClassifier:
    """
    Binary classification model for churn risk prediction.
    
    Churn Definition:
    Customer is churned if Recency > churn_threshold days (default: 90)
    """
    
    def __init__(self, churn_threshold_days: int = 90, random_state: int = 42):
        """
        Initialize churn classifier.
        
        Args:
            churn_threshold_days: Days of inactivity to define churn (default: 90)
            random_state: Random seed for reproducibility
        """
        self.churn_threshold = churn_threshold_days
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.optimal_threshold = 0.5
        
    def prepare_features(self, rfm: pd.DataFrame, transactions: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare features for churn prediction.
        
        Args:
            rfm: DataFrame with RFM metrics
            transactions: Optional transaction data for additional features
            
        Returns:
            DataFrame with features and churn labels
        """
        logger.info("Preparing features for churn classification")
        
        df = rfm.copy()
        
        # Create churn label (target variable)
        df['Churn'] = (df['Recency'] > self.churn_threshold).astype(int)
        
        churn_count = df['Churn'].sum()
        churn_rate = (churn_count / len(df)) * 100
        
        logger.info(f"Churn definition: Recency > {self.churn_threshold} days")
        logger.info(f"Churned customers: {churn_count:,} ({churn_rate:.1f}%)")
        logger.info(f"Active customers: {len(df) - churn_count:,} ({100-churn_rate:.1f}%)")
        
        # Feature engineering
        df['Recency_Frequency_Ratio'] = df['Recency'] / (df['Frequency'] + 1)
        df['Monetary_Frequency_Ratio'] = df['Monetary'] / (df['Frequency'] + 1)
        df['Avg_Order_Value'] = df['Monetary'] / (df['Frequency'] + 1)
        
        # Interaction features
        df['RFM_Interaction'] = df['Recency'] * df['Frequency'] * df['Monetary']
        df['RF_Product'] = df['Recency'] * df['Frequency']
        df['RM_Product'] = df['Recency'] * df['Monetary']
        df['FM_Product'] = df['Frequency'] * df['Monetary']
        
        logger.info(f"Features created: {len(df.columns)} total columns")
        
        return df
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        """
        Train churn classification model.
        
        Args:
            df: DataFrame with features and Churn label
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training churn classification model")
        
        # Prepare features and target
        feature_cols = ['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score',
                       'Recency_Frequency_Ratio', 'Monetary_Frequency_Ratio', 'Avg_Order_Value',
                       'RF_Product', 'RM_Product', 'FM_Product']
        
        X = df[feature_cols]
        y = df['Churn']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Handle class imbalance with SMOTE
        logger.info("Applying SMOTE for class imbalance handling")
        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE: {len(X_train_balanced):,} samples")
        logger.info(f"Class distribution: 0={sum(y_train_balanced==0):,}, 1={sum(y_train_balanced==1):,}")
        
        # Train LightGBM
        logger.info("Training LightGBM classifier")
        
        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=200,
            random_state=self.random_state,
            verbose=-1
        )
        
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Evaluation
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        logger.info("Model training complete")
        
        return metrics
    
    def _evaluate_model(self, y_test, y_pred, y_pred_proba) -> dict:
        """Calculate model performance metrics."""
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"ROC-AUC Score: {roc_auc:.3f}")
        logger.info(f"Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate churn predictions for all customers.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with churn predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        logger.info("Generating churn predictions for all customers")
        
        feature_cols = ['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score',
                       'Recency_Frequency_Ratio', 'Monetary_Frequency_Ratio', 'Avg_Order_Value',
                       'RF_Product', 'RM_Product', 'FM_Product']
        
        X = df[feature_cols]
        
        # Predict probabilities
        churn_proba = self.model.predict_proba(X)[:, 1]
        churn_label = (churn_proba >= self.optimal_threshold).astype(int)
        
        # Create results
        results = df[['CustomerID', 'Recency', 'Frequency', 'Monetary']].copy()
        results['Churn_Probability'] = churn_proba
        results['Churn_Predicted'] = churn_label
        results['Actual_Churn'] = df['Churn']
        
        # Risk levels
        results['Risk_Level'] = pd.cut(
            results['Churn_Probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        logger.info(f"Predictions generated for {len(results):,} customers")
        logger.info(f"High risk (>0.7): {(results['Churn_Probability'] > 0.7).sum():,}")
        logger.info(f"Medium risk (0.3-0.7): {((results['Churn_Probability'] >= 0.3) & (results['Churn_Probability'] <= 0.7)).sum():,}")
        logger.info(f"Low risk (<0.3): {(results['Churn_Probability'] < 0.3).sum():,}")
        
        return results


def main():
    """Execute churn classification pipeline."""
    
    print("="*80)
    print("CHURN RISK CLASSIFICATION")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load RFM data
    rfm_path = Path("data/features/customer_rfm.csv")
    
    if not rfm_path.exists():
        logger.error(f"RFM data not found: {rfm_path}")
        logger.error("Run src/data/feature_engineering.py first")
        return
    
    logger.info(f"Loading RFM data from: {rfm_path}")
    rfm = pd.read_csv(rfm_path)
    logger.info(f"Loaded RFM for {len(rfm):,} customers")
    
    # Initialize classifier
    classifier = ChurnClassifier(churn_threshold_days=90, random_state=42)
    
    # Prepare features
    df_features = classifier.prepare_features(rfm)
    
    # Train model
    print("\n" + "-"*80)
    metrics = classifier.train(df_features, test_size=0.2)
    
    # Generate predictions for all customers
    print("\n" + "-"*80)
    churn_predictions = classifier.predict(df_features)
    
    # Display results
    print("\n" + "="*80)
    print("CHURN PREDICTION RESULTS")
    print("="*80)
    
    print("\nRisk Distribution:")
    print(churn_predictions['Risk_Level'].value_counts())
    
    print("\nHigh-Risk Customers (Probability > 0.7):")
    high_risk = churn_predictions[churn_predictions['Churn_Probability'] > 0.7]
    print(f"Count: {len(high_risk):,}")
    
    if len(high_risk) > 0:
        print("\nTop 10 Highest Risk:")
        print(high_risk.head(10)[['CustomerID', 'Churn_Probability', 'Recency', 
                                  'Frequency', 'Monetary']].to_string(index=False))
    
    print("\nFeature Importance:")
    print(classifier.feature_importance.head(10).to_string(index=False))
    
    # Calculate at-risk revenue (merge with CLV)
    clv_path = Path("data/predictions/customer_clv.csv")
    if clv_path.exists():
        clv = pd.read_csv(clv_path)
        churn_with_clv = churn_predictions.merge(clv[['CustomerID', 'Predicted_CLV_12m']], on='CustomerID')
        
        high_risk_revenue = churn_with_clv[churn_with_clv['Churn_Probability'] > 0.7]['Predicted_CLV_12m'].sum()
        
        print(f"\nAt-Risk Revenue Analysis:")
        print(f"High-risk customers: {len(high_risk):,}")
        print(f"At-risk predicted revenue: ${high_risk_revenue:,.2f}")
    
    # Save predictions
    output_path = Path("data/predictions/customer_churn.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    churn_predictions.to_csv(output_path, index=False)
    
    logger.info(f"Churn predictions saved to: {output_path}")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    try:
        with open(models_dir / "churn_model.pkl", 'wb') as f:
            dill.dump(classifier.model, f)
        logger.info("Churn model saved: models/churn_model.pkl")
    except Exception as e:
        logger.warning(f"Could not save model: {e}")
    
    # Save metrics
    with open(models_dir / "churn_model_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Model metrics saved: models/churn_model_metrics.json")
    
    # Save feature importance
    classifier.feature_importance.to_csv(models_dir / "churn_feature_importance.csv", index=False)
    logger.info("Feature importance saved: models/churn_feature_importance.csv")
    
    print("\n" + "="*80)
    print("CHURN CLASSIFICATION COMPLETE")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Next step: Combine all predictions into master file")


if __name__ == "__main__":
    main()
