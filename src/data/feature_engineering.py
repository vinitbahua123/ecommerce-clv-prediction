"""
Feature Engineering Module - RFM Metric Calculation
Professional implementation with validation and audit trail


Date: February 2026

Purpose:
Calculate Recency, Frequency, and Monetary metrics for customer segmentation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RFMCalculator:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics.
    
    Methodology:
    - Recency: Days since last purchase (lower is better)
    - Frequency: Number of unique transactions (higher is better)
    - Monetary: Total revenue generated (higher is better)
    """
    
    def __init__(self, reference_date=None):
        """
        Initialize RFM calculator.
        
        Args:
            reference_date: Reference date for recency calculation
                          (default: max date in data + 1 day)
        """
        self.reference_date = reference_date
        self.rfm_stats = {}
    
    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.
        
        Args:
            df: Cleaned transaction DataFrame with columns:
                - CustomerID
                - InvoiceDate
                - InvoiceNo
                - TotalAmount
                
        Returns:
            DataFrame with CustomerID and RFM metrics
        """
        logger.info("Calculating RFM metrics for customer base")
        
        # Validate required columns
        required_cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalAmount']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Set reference date
        if self.reference_date is None:
            max_date = pd.to_datetime(df['InvoiceDate']).max()
            self.reference_date = max_date + timedelta(days=1)
        
        logger.info(f"Reference date for recency: {self.reference_date}")
        
        # Calculate RFM metrics
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (self.reference_date - pd.to_datetime(x).max()).days,
            'InvoiceNo': 'nunique',
            'TotalAmount': 'sum'
        }).reset_index()
        
        # Rename columns
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Validation
        self._validate_rfm(rfm)
        
        # Store statistics
        self._calculate_statistics(rfm)
        
        logger.info(f"RFM calculation complete for {len(rfm):,} customers")
        
        return rfm
    
    def _validate_rfm(self, rfm: pd.DataFrame):
        """Validate RFM calculations."""
        assert rfm['Recency'].min() >= 0, "Recency cannot be negative"
        assert rfm['Frequency'].min() >= 1, "Frequency must be at least 1"
        assert rfm['Monetary'].min() > 0, "Monetary must be positive"
        
        logger.info("RFM validation passed")
    
    def _calculate_statistics(self, rfm: pd.DataFrame):
        """Calculate and log RFM statistics."""
        self.rfm_stats = {
            'recency_min': rfm['Recency'].min(),
            'recency_max': rfm['Recency'].max(),
            'recency_mean': rfm['Recency'].mean(),
            'recency_median': rfm['Recency'].median(),
            'frequency_min': rfm['Frequency'].min(),
            'frequency_max': rfm['Frequency'].max(),
            'frequency_mean': rfm['Frequency'].mean(),
            'frequency_median': rfm['Frequency'].median(),
            'monetary_min': rfm['Monetary'].min(),
            'monetary_max': rfm['Monetary'].max(),
            'monetary_mean': rfm['Monetary'].mean(),
            'monetary_median': rfm['Monetary'].median()
        }
        
        logger.info(f"Recency range: {self.rfm_stats['recency_min']:.0f} to {self.rfm_stats['recency_max']:.0f} days")
        logger.info(f"Recency mean: {self.rfm_stats['recency_mean']:.1f} days, median: {self.rfm_stats['recency_median']:.0f} days")
        logger.info(f"Frequency range: {self.rfm_stats['frequency_min']:.0f} to {self.rfm_stats['frequency_max']:.0f} orders")
        logger.info(f"Frequency mean: {self.rfm_stats['frequency_mean']:.1f} orders, median: {self.rfm_stats['frequency_median']:.0f} orders")
        logger.info(f"Monetary range: ${self.rfm_stats['monetary_min']:.2f} to ${self.rfm_stats['monetary_max']:,.2f}")
        logger.info(f"Monetary mean: ${self.rfm_stats['monetary_mean']:,.2f}, median: ${self.rfm_stats['monetary_median']:.2f}")
    
    def calculate_rfm_scores(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate quintile-based RFM scores (1-5 scale).
        
        Scoring methodology:
        - Divide customers into 5 equal groups (quintiles)
        - Recency: Lower values get higher scores (recent = better)
        - Frequency: Higher values get higher scores (frequent = better)
        - Monetary: Higher values get higher scores (high spend = better)
        
        Args:
            rfm: DataFrame with Recency, Frequency, Monetary columns
            
        Returns:
            DataFrame with added score columns
        """
        logger.info("Calculating quintile-based RFM scores")
        
        rfm_scored = rfm.copy()
        
        # Recency Score (inverse: lower recency = higher score)
        rfm_scored['R_Score'] = pd.qcut(
            rfm['Recency'],
            q=5,
            labels=[5, 4, 3, 2, 1],
            duplicates='drop'
        ).astype(int)
        
        # Frequency Score (direct: higher frequency = higher score)
        rfm_scored['F_Score'] = pd.qcut(
            rfm['Frequency'].rank(method='first'),
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(int)
        
        # Monetary Score (direct: higher monetary = higher score)
        rfm_scored['M_Score'] = pd.qcut(
            rfm['Monetary'].rank(method='first'),
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(int)
        
        # Combined RFM Score (concatenated string)
        rfm_scored['RFM_Score'] = (
            rfm_scored['R_Score'].astype(str) +
            rfm_scored['F_Score'].astype(str) +
            rfm_scored['M_Score'].astype(str)
        )
        
        # Numeric average for alternative scoring
        rfm_scored['RFM_Score_Avg'] = (
            rfm_scored['R_Score'] + 
            rfm_scored['F_Score'] + 
            rfm_scored['M_Score']
        ) / 3
        
        logger.info("RFM scores calculated")
        logger.info(f"Score distribution - R: {rfm_scored['R_Score'].value_counts().sort_index().to_dict()}")
        logger.info(f"Score distribution - F: {rfm_scored['F_Score'].value_counts().sort_index().to_dict()}")
        logger.info(f"Score distribution - M: {rfm_scored['M_Score'].value_counts().sort_index().to_dict()}")
        
        return rfm_scored


def main():
    """Execute RFM calculation pipeline."""
    
    print("="*80)
    print("RFM METRIC CALCULATION")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load cleaned data
    input_path = Path("data/processed/online_retail_clean.csv")
    
    if not input_path.exists():
        logger.error(f"Cleaned data not found: {input_path}")
        logger.error("Please run src/data/clean_data.py first")
        return
    
    logger.info(f"Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=['InvoiceDate'])
    logger.info(f"Loaded: {len(df):,} transactions")
    logger.info(f"Unique customers: {df['CustomerID'].nunique():,}")
    
    # Initialize calculator
    calculator = RFMCalculator()
    
    # Calculate RFM
    rfm = calculator.calculate_rfm(df)
    
    # Add scores
    rfm_scored = calculator.calculate_rfm_scores(rfm)
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE RFM DATA (First 10 Customers)")
    print("="*80)
    print(rfm_scored.head(10).to_string(index=False))
    
    # Display statistics
    print("\n" + "="*80)
    print("RFM STATISTICS")
    print("="*80)
    print("\nRecency (Days since last purchase):")
    print(rfm['Recency'].describe())
    print("\nFrequency (Number of orders):")
    print(rfm['Frequency'].describe())
    print("\nMonetary (Total spend):")
    print(rfm['Monetary'].describe())
    
    # Save RFM data
    output_path = Path("data/features/customer_rfm.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rfm_scored.to_csv(output_path, index=False)
    
    logger.info(f"RFM data saved to: {output_path}")
    logger.info(f"Customers: {len(rfm_scored):,}")
    logger.info(f"Columns: {list(rfm_scored.columns)}")
    
    print("\n" + "="*80)
    print("RFM CALCULATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
