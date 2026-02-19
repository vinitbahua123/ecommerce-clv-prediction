"""
Customer Lifetime Value Prediction Module
Implementation of BG/NBD and Gamma-Gamma models

Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json

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
        """
        Initialize CLV predictor.
        
        Args:
            prediction_period_months: Forecast horizon (default: 12 months)
            discount_rate: Monthly discount rate (default: 0.01)
        """
        self.prediction_period = prediction_period_months
        self.discount_rate = discount_rate
        self.bgf_model = None
        self.ggf_model = None
        
    def prepare_summary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare customer summary for lifetimes models."""
        logger.info("Preparing transaction data for CLV modeling")
        
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        observation_end = df['InvoiceDate'].max()
        logger.info(f"Observation period end date: {observation_end}")
        
        summary = summary_data_from_transaction_data(
            df,
            customer_id_col='CustomerID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalAmount',
            observation_period_end=observation_end
        )
        
        logger.info(f"Summary prepared for {len(summary):,} customers")
        logger.info(f"Frequency range: {summary['frequency'].min():.0f} to {summary['frequency'].max():.0f}")
        logger.info(f"Recency range: {summary['recency'].min():.0f} to {summary['recency'].max():.0f}")
        logger.info(f"T (customer age) range: {summary['T'].min():.0f} to {summary['T'].max():.0f}")
        logger.info(f"Monetary range: ${summary['monetary_value'].min():.2f} to ${summary['monetary_value'].max():,.2f}")
        
        return summary
    
    def train_bgf_model(self, summary: pd.DataFrame) -> BetaGeoFitter:
        """Train BG/NBD model for purchase frequency prediction."""
        logger.info("Training BG/NBD model for purchase frequency prediction")
        
        self.bgf_model = BetaGeoFitter(penalizer_coef=0.0)
        self.bgf_model.fit(summary['frequency'], summary['recency'], summary['T'])
        
        logger.info("BG/NBD model training complete")
        logger.info(f"Parameters: r={self.bgf_model.params_['r']:.4f}, "
                   f"alpha={self.bgf_model.params_['alpha']:.4f}, "
                   f"a={self.bgf_model.params_['a']:.4f}, "
                   f"b={self.bgf_model.params_['b']:.4f}")
        
        return self.bgf_model
    
    def train_ggf_model(self, summary: pd.DataFrame) -> GammaGammaFitter:
        """Train Gamma-Gamma model for monetary value prediction."""
        logger.info("Training Gamma-Gamma model for monetary value prediction")
        
        returning = summary[summary['frequency'] > 0]
        logger.info(f"Repeat customers for training: {len(returning):,} of {len(summary):,}")
        
        self.ggf_model = GammaGammaFitter(penalizer_coef=0.0)
        self.ggf_model.fit(returning['frequency'], returning['monetary_value'])
        
        logger.info("Gamma-Gamma model training complete")
        logger.info(f"Parameters: p={self.ggf_model.params_['p']:.4f}, "
                   f"q={self.ggf_model.params_['q']:.4f}, "
                   f"v={self.ggf_model.params_['v']:.4f}")
        
        return self.ggf_model
    
    def predict_clv(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Generate CLV predictions for all customers."""
        if self.bgf_model is None or self.ggf_model is None:
            raise ValueError("Models not trained")
        
        logger.info(f"Calculating {self.prediction_period}-month CLV predictions")
        
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
            'Customer_Age_Days': summary['T'],
            'Expected_Purchases_12m': self.bgf_model.predict(
                self.prediction_period, summary['frequency'], 
                summary['recency'], summary['T']
            ),
            'Probability_Alive': self.bgf_model.conditional_probability_alive(
                summary['frequency'], summary['recency'], summary['T']
            )
        })
        
        results['Historical_Total_Value'] = (summary['frequency'] + 1) * summary['monetary_value']
        results = results.sort_values('Predicted_CLV_12m', ascending=False).reset_index(drop=True)
        
        logger.info("CLV predictions generated")
        logger.info(f"Total predicted revenue (12 months): ${results['Predicted_CLV_12m'].sum():,.2f}")
        logger.info(f"Average predicted CLV: ${results['Predicted_CLV_12m'].mean():,.2f}")
        logger.info(f"Median predicted CLV: ${results['Predicted_CLV_12m'].median():,.2f}")
        
        return results
    
    def save_model_parameters(self, output_dir: Path):
        """
        Save model parameters as JSON (Python 3.13 compatible).
        
        Args:
            output_dir: Directory to save parameters
        """
        if self.bgf_model is None or self.ggf_model is None:
            logger.warning("Models not trained, cannot save parameters")
            return
        
        # BG/NBD parameters
        bgf_params = {
            'model_type': 'BetaGeoFitter',
            'parameters': {
                'r': float(self.bgf_model.params_['r']),
                'alpha': float(self.bgf_model.params_['alpha']),
                'a': float(self.bgf_model.params_['a']),
                'b': float(self.bgf_model.params_['b'])
            },
            'penalizer_coef': 0.0,
            'trained_date': datetime.now().isoformat()
        }
        
        # Gamma-Gamma parameters
        ggf_params = {
            'model_type': 'GammaGammaFitter',
            'parameters': {
                'p': float(self.ggf_model.params_['p']),
                'q': float(self.ggf_model.params_['q']),
                'v': float(self.ggf_model.params_['v'])
            },
            'penalizer_coef': 0.0,
            'trained_date': datetime.now().isoformat()
        }
        
        # Save parameters
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'bgf_model_params.json', 'w') as f:
            json.dump(bgf_params, f, indent=2)
        
        with open(output_dir / 'ggf_model_params.json', 'w') as f:
            json.dump(ggf_params, f, indent=2)
        
        logger.info(f"Model parameters saved to: {output_dir}/")
        logger.info("Note: Full model objects not saved due to Python 3.13 serialization issues")


def main():
    """Execute CLV prediction pipeline."""
    
    print("="*80)
    print("CUSTOMER LIFETIME VALUE PREDICTION")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load cleaned data
    input_path = Path("data/processed/online_retail_clean.csv")
    
    if not input_path.exists():
        logger.error(f"Cleaned data not found: {input_path}")
        logger.error("Run src/data/clean_data.py first")
        return
    
    logger.info(f"Loading transaction data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=['InvoiceDate'])
    logger.info(f"Loaded: {len(df):,} transactions from {df['CustomerID'].nunique():,} customers")
    logger.info(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    
    # Initialize predictor
    predictor = CLVPredictor(prediction_period_months=12, discount_rate=0.01)
    
    # Prepare data
    summary = predictor.prepare_summary_data(df)
    
    # Train models
    print("\n" + "-"*80)
    bgf_model = predictor.train_bgf_model(summary)
    
    print("\n" + "-"*80)
    ggf_model = predictor.train_ggf_model(summary)
    
    # Generate predictions
    print("\n" + "-"*80)
    clv_predictions = predictor.predict_clv(summary)
    
    # Display results
    print("\n" + "="*80)
    print("CLV PREDICTION RESULTS")
    print("="*80)
    
    print("\nTop 10 Customers by Predicted 12-Month CLV:")
    top10 = clv_predictions.head(10)[['CustomerID', 'Predicted_CLV_12m', 
                                      'Historical_Total_Value', 'Probability_Alive']]
    print(top10.to_string(index=False))
    
    print("\nPredicted CLV Distribution:")
    print(clv_predictions['Predicted_CLV_12m'].describe())
    
    # Revenue concentration
    top_20pct = clv_predictions.head(int(len(clv_predictions) * 0.2))
    top_20_revenue = top_20pct['Predicted_CLV_12m'].sum()
    total_revenue = clv_predictions['Predicted_CLV_12m'].sum()
    concentration = (top_20_revenue / total_revenue) * 100
    
    print(f"\nRevenue Concentration Analysis:")
    print(f"Top 20% of customers predicted to generate: {concentration:.1f}% of revenue")
    print(f"Top 20% predicted revenue: ${top_20_revenue:,.2f}")
    print(f"Total predicted revenue (12 months): ${total_revenue:,.2f}")
    
    # Save predictions
    output_path = Path("data/predictions/customer_clv.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clv_predictions.to_csv(output_path, index=False)
    
    logger.info(f"CLV predictions saved to: {output_path}")
    logger.info(f"Total customers with predictions: {len(clv_predictions):,}")
    
    # Save model parameters (Python 3.13 compatible)
    models_dir = Path("models")
    predictor.save_model_parameters(models_dir)
    
    print("\n" + "="*80)
    print("CLV PREDICTION COMPLETE")
    print("="*80)
    print(f"\nOutput file: {output_path}")
    print(f"Model parameters: {models_dir}/bgf_model_params.json")
    print(f"                  {models_dir}/ggf_model_params.json")
    print(f"\nNext step: Churn risk classification")


if __name__ == "__main__":
    main()
