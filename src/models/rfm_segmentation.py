"""
Customer Segmentation Module - K-Means Clustering
Professional implementation with business-focused tier naming


Date: February 2026
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path
from datetime import datetime
import logging
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CustomerSegmentation:
    """K-Means clustering for customer segmentation using RFM metrics."""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize segmentation model.
        
        Args:
            n_clusters: Number of customer segments
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = None
        
    def fit_predict(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Train K-Means model and assign segment labels.
        
        Args:
            rfm: DataFrame with Recency, Frequency, Monetary columns
            
        Returns:
            DataFrame with cluster and segment assignments
        """
        logger.info(f"Training K-Means clustering model with {self.n_clusters} clusters")
        
        # Prepare feature matrix
        X = rfm[['Recency', 'Frequency', 'Monetary']].values
        
        # Scale features (critical for K-Means)
        X_scaled = self.scaler.fit_transform(X)
        logger.info("Features scaled using StandardScaler")
        
        # Train K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=self.random_state
        )
        
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Evaluate clustering quality
        silhouette = silhouette_score(X_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
        
        logger.info(f"K-Means training complete")
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
        
        # Add cluster labels
        rfm_clustered = rfm.copy()
        rfm_clustered['Cluster'] = cluster_labels
        
        # Assign business-relevant segment names
        rfm_clustered = self._assign_segment_names(rfm_clustered)
        
        return rfm_clustered
    
    def _assign_segment_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign tier-based names to clusters based on characteristics.
        
        Args:
            df: DataFrame with Cluster column
            
        Returns:
            DataFrame with Segment column added
        """
        logger.info("Assigning segment names based on cluster characteristics")
        
        # Analyze each cluster
        cluster_profiles = df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count'
        })
        
        # Assign names based on data-driven analysis
        segment_mapping = {}
        
        for cluster_id in cluster_profiles.index:
            r_avg = cluster_profiles.loc[cluster_id, 'Recency']
            f_avg = cluster_profiles.loc[cluster_id, 'Frequency']
            m_avg = cluster_profiles.loc[cluster_id, 'Monetary']
            
            # Tier assignment logic
            if m_avg > 50000:
                segment_mapping[cluster_id] = 'Platinum Tier'
            elif m_avg > 10000 and f_avg > 15:
                segment_mapping[cluster_id] = 'Gold Tier'
            elif r_avg < 100 and f_avg > 2:
                segment_mapping[cluster_id] = 'Silver Tier'
            elif r_avg > 200:
                segment_mapping[cluster_id] = 'Inactive'
            else:
                segment_mapping[cluster_id] = 'Bronze Tier'
        
        df['Segment'] = df['Cluster'].map(segment_mapping)
        
        # Log segment distribution
        logger.info("Segment distribution:")
        for segment in sorted(df['Segment'].unique()):
            count = (df['Segment'] == segment).sum()
            pct = (count / len(df)) * 100
            logger.info(f"  {segment}: {count:,} customers ({pct:.1f}%)")
        
        return df
    
    def create_segment_profiles(self, rfm_segmented: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical profiles for each segment.
        
        Args:
            rfm_segmented: DataFrame with Segment column
            
        Returns:
            DataFrame with segment-level statistics
        """
        logger.info("Creating segment profiles")
        
        profiles = rfm_segmented.groupby('Segment').agg({
            'CustomerID': 'count',
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'],
            'Monetary': ['mean', 'median', 'sum']
        }).round(2)
        
        # Flatten multi-level columns
        profiles.columns = ['_'.join(col).strip() for col in profiles.columns.values]
        profiles = profiles.reset_index()
        
        # Calculate percentages
        total_customers = profiles['CustomerID_count'].sum()
        profiles['Customer_Percentage'] = (profiles['CustomerID_count'] / total_customers * 100).round(1)
        
        total_revenue = profiles['Monetary_sum'].sum()
        profiles['Revenue_Percentage'] = (profiles['Monetary_sum'] / total_revenue * 100).round(1)
        
        # Rename for clarity
        profiles.rename(columns={
            'CustomerID_count': 'Customer_Count',
            'Recency_mean': 'Avg_Recency_Days',
            'Recency_median': 'Median_Recency_Days',
            'Frequency_mean': 'Avg_Frequency',
            'Frequency_median': 'Median_Frequency',
            'Monetary_mean': 'Avg_Monetary_Value',
            'Monetary_median': 'Median_Monetary_Value',
            'Monetary_sum': 'Total_Revenue'
        }, inplace=True)
        
        # Sort by revenue contribution
        profiles = profiles.sort_values('Revenue_Percentage', ascending=False)
        
        return profiles


def main():
    """Execute customer segmentation pipeline."""
    
    print("="*80)
    print("CUSTOMER SEGMENTATION")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load pre-calculated RFM data
    rfm_path = Path("data/features/customer_rfm.csv")
    
    if not rfm_path.exists():
        logger.error(f"RFM data not found: {rfm_path}")
        logger.error("Please run src/data/feature_engineering.py first")
        return
    
    logger.info(f"Loading RFM data from: {rfm_path}")
    rfm = pd.read_csv(rfm_path)
    logger.info(f"Loaded RFM for {len(rfm):,} customers")
    
    # Perform segmentation
    segmenter = CustomerSegmentation(n_clusters=5, random_state=42)
    rfm_segmented = segmenter.fit_predict(rfm)
    
    # Create profiles
    profiles = segmenter.create_segment_profiles(rfm_segmented)
    
    # Display profiles
    print("\n" + "="*80)
    print("CUSTOMER SEGMENT PROFILES")
    print("="*80)
    print(profiles.to_string(index=False))
    
    # Business insights
    print("\n" + "="*80)
    print("BUSINESS INSIGHTS")
    print("="*80)
    
    for _, row in profiles.iterrows():
        print(f"\n{row['Segment']}:")
        print(f"  Customers: {row['Customer_Count']:,} ({row['Customer_Percentage']:.1f}%)")
        print(f"  Revenue: ${row['Total_Revenue']:,.2f} ({row['Revenue_Percentage']:.1f}%)")
        print(f"  Avg Value: ${row['Avg_Monetary_Value']:,.2f}")
        print(f"  Avg Recency: {row['Avg_Recency_Days']:.0f} days")
        print(f"  Avg Frequency: {row['Avg_Frequency']:.1f} orders")
    
    # Save results
    segments_output = Path("data/features/customer_segments.csv")
    rfm_segmented.to_csv(segments_output, index=False)
    
    profiles_output = Path("data/features/segment_profiles.csv")
    profiles.to_csv(profiles_output, index=False)
    
    logger.info(f"Segmentation saved to: {segments_output}")
    logger.info(f"Profiles saved to: {profiles_output}")
    
    # Save models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(segmenter.kmeans, models_dir / "kmeans_model.pkl")
    joblib.dump(segmenter.scaler, models_dir / "scaler_rfm.pkl")
    
    logger.info(f"Models saved to: {models_dir}/")
    
    print("\n" + "="*80)
    print("SEGMENTATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
