"""
Customer Lifetime Value Prediction Module
Implementation of BG/NBD and Gamma-Gamma models

Author: CLV Analytics Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import dill  # More powerful than pickle for Python 3.13

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CLVPredictor:
    """Customer Lifetime Value prediction using statistical models."""
    
    def __init__(self, prediction_period_months: int = 12, discount_rate: float = 0.01):
        self.prediction_period = prediction_period_months
        self.discount_rate = discount_rate
        self.bgf_model = None
        self.ggf_model = None
        
    def prepare_summary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare customer summary for lifetimes models."""
        logger.info("Preparing transaction data for CLV modeling")
        
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        observation_end = df['InvoiceDate'].max()
        logger.info(f"Observation period end: {observation_end}")
        
        summary = summary_data_from_transaction_data(
            df,
            customer_id_col='CustomerID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalAmount',
            observation_period_end=observation_end
        )
        
        logger.info(f"Summary prepared for {len(summary):,} customers")
        return summary
    
    def train_bgf_model(self, summary: pd.DataFrame) -> BetaGeoFitter:
        """Train BG/NBD model."""
        logger.info("Training BG/NBD model")
        
        self.bgf_model = BetaGeoFitter(penalizer_coef=0.0)
        self.bgf_model.fit(summary['frequency'], summary['recency'], summary['T'])
        
        logger.info("BG/NBD training complete")
        logger.info(f"Parameters: r={self.bgf_model.params_['r']:.4f}, "
                   f"alpha={self.bgf_model.params_['alpha']:.4f}")
        return self.bgf_model
    
    def train_ggf_model(self, summary: pd.DataFrame) -> GammaGammaFitter:
        """Train Gamma-Gamma model."""
        logger.info("Training Gamma-Gamma model")
        
        returning = summary[summary['frequency'] > 0]
        logger.info(f"Repeat customers: {len(returning):,} of {len(summary):,}")
        
        self.ggf_model = GammaGammaFitter(penalizer_coef=0.0)
        self.ggf_model.fit(returning['frequency'], returning['monetary_value'])
        
        logger.info("Gamma-Gamma training complete")
        return self.ggf_model
    
    def predict_clv(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Generate CLV predictions."""
        logger.info(f"Calculating {self.prediction_period}-month CLV")
        
        clv_values = self.ggf_model.customer_lifetime_value(
            self.bgf_model,
            summary['frequency'],
            summary['recency'],
            summary['T'],
            summary['monetary_value'],
            time=self.prediction_period,
            discount_rate=self.discount_rate
        )
        
        results = pd.DataFrame({
            'CustomerID': summary.index,
            'Predicted_CLV_12m': clv_values,
            'Historical_Frequency': summary['frequency'],
            'Historical_Recency': summary['recency'],
            'Historical_Monetary_Avg': summary['monetary_value'],
            'Expected_Purchases_12m': self.bgf_model.predict(
                self.prediction_period, summary['frequency'], 
                summary['recency'], summary['T']
            ),
            'Probability_Alive': self.bgf_model.conditional_probability_alive(
                summary['frequency'], summary['recency'], summary['T']
            )
        })
        
        results = results.sort_values('Predicted_CLV_12m', ascending=False).reset_index(drop=True)
        
        logger.info(f"Total predicted revenue: ${results['Predicted_CLV_12m'].sum():,.2f}")
        logger.info(f"Average CLV: ${results['Predicted_CLV_12m'].mean():,.2f}")
        
        return results


def main():
    """Execute CLV prediction pipeline."""
    
    print("="*80)
    print("CUSTOMER LIFETIME VALUE PREDICTION")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    df = pd.read_csv("data/processed/online_retail_clean.csv", parse_dates=['InvoiceDate'])
    logger.info(f"Loaded: {len(df):,} transactions, {df['CustomerID'].nunique():,} customers")
    
    # Run pipeline
    predictor = CLVPredictor(prediction_period_months=12)
    summary = predictor.prepare_summary_data(df)
    
    print("\n" + "-"*80)
    predictor.train_bgf_model(summary)
    
    print("\n" + "-"*80)
    predictor.train_ggf_model(summary)
    
    print("\n" + "-"*80)
    clv_predictions = predictor.predict_clv(summary)
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nTop 10 Customers:")
    print(clv_predictions.head(10)[['CustomerID', 'Predicted_CLV_12m', 'Probability_Alive']].to_string(index=False))
    
    print("\nCLV Distribution:")
    print(clv_predictions['Predicted_CLV_12m'].describe())
    
    top20 = clv_predictions.head(int(len(clv_predictions) * 0.2))
    pct = (top20['Predicted_CLV_12m'].sum() / clv_predictions['Predicted_CLV_12m'].sum()) * 100
    print(f"\nTop 20%: {pct:.1f}% of predicted revenue")
    
    # Save predictions
    output = Path("data/predictions/customer_clv.csv")
    output.parent.mkdir(exist_ok=True)
    clv_predictions.to_csv(output, index=False)
    logger.info(f"Predictions saved: {output}")
    
    # Save models using dill (Python 3.13 compatible)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    try:
        with open(models_dir / "bgf_model.pkl", 'wb') as f:
            dill.dump(predictor.bgf_model, f)
        logger.info("BG/NBD model saved: models/bgf_model.pkl")
        
        with open(models_dir / "ggf_model.pkl", 'wb') as f:
            dill.dump(predictor.ggf_model, f)
        logger.info("Gamma-Gamma model saved: models/ggf_model.pkl")
        
    except Exception as e:
        logger.warning(f"Could not save model objects: {e}")
        logger.info("Models can be retrained from data if needed")
    
    print("\n" + "="*80)
    print("CLV PREDICTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
