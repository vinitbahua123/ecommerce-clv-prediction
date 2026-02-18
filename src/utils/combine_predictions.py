"""
Combine All Predictions into Master File
Merges segmentation, CLV, and churn predictions

Author: CLV Analytics Team
Date: February 2026
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def combine_all_predictions():
    """Merge all prediction files into single master dataset."""
    
    print("="*80)
    print("COMBINING ALL PREDICTIONS INTO MASTER FILE")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load all prediction files
    logger.info("Loading prediction files")
    
    segments = pd.read_csv("data/features/customer_segments.csv")
    logger.info(f"Loaded segments: {len(segments):,} customers")
    
    clv = pd.read_csv("data/predictions/customer_clv.csv")
    logger.info(f"Loaded CLV: {len(clv):,} customers")
    
    churn = pd.read_csv("data/predictions/customer_churn.csv")
    logger.info(f"Loaded churn: {len(churn):,} customers")
    
    # Merge all on CustomerID
    logger.info("Merging datasets on CustomerID")
    
    master = segments.merge(clv[['CustomerID', 'Predicted_CLV_12m', 'Expected_Purchases_12m', 
                                 'Probability_Alive']], 
                           on='CustomerID', how='left')
    
    master = master.merge(churn[['CustomerID', 'Churn_Probability', 'Risk_Level']], 
                         on='CustomerID', how='left')
    
    logger.info(f"Master dataset created: {len(master):,} customers, {len(master.columns)} columns")
    
    # Add business recommendations
    logger.info("Adding business recommendations")
    
    def assign_recommendation(row):
        """Assign action based on segment and risk."""
        if row['Segment'] == 'Platinum Tier':
            if row['Risk_Level'] == 'High':
                return 'URGENT: Personal call from account manager'
            else:
                return 'Maintain: Continue white-glove service'
        
        elif row['Segment'] == 'Gold Tier':
            if row['Risk_Level'] == 'High':
                return 'High Priority: Send retention offer'
            elif row['Risk_Level'] == 'Medium':
                return 'Monitor: Proactive engagement'
            else:
                return 'Maintain: VIP loyalty program'
        
        elif row['Segment'] == 'Silver Tier':
            if row['Risk_Level'] == 'High':
                return 'Send: Standard win-back offer'
            elif row['Risk_Level'] == 'Medium':
                return 'Engage: Regular communication'
            else:
                return 'Maintain: Standard loyalty program'
        
        else:  # Inactive
            return 'Low Priority: Automated email only'
    
    master['Recommended_Action'] = master.apply(assign_recommendation, axis=1)
    
    # Reorder columns for clarity
    column_order = [
        'CustomerID',
        'Segment',
        'Cluster',
        'Recency',
        'Frequency',
        'Monetary',
        'R_Score',
        'F_Score',
        'M_Score',
        'RFM_Score',
        'Predicted_CLV_12m',
        'Expected_Purchases_12m',
        'Probability_Alive',
        'Churn_Probability',
        'Risk_Level',
        'Recommended_Action'
    ]
    
    master = master[column_order]
    
    # Display summary statistics
    print("\n" + "="*80)
    print("MASTER DATASET SUMMARY")
    print("="*80)
    
    print(f"\nTotal customers: {len(master):,}")
    print(f"Total columns: {len(master.columns)}")
    
    print("\nSegment Distribution:")
    print(master['Segment'].value_counts())
    
    print("\nRisk Distribution:")
    print(master['Risk_Level'].value_counts())
    
    print("\nTop 10 Customers (by Predicted CLV):")
    top10 = master.nlargest(10, 'Predicted_CLV_12m')[['CustomerID', 'Segment', 
                                                       'Predicted_CLV_12m', 'Risk_Level']]
    print(top10.to_string(index=False))
    
    # High-risk high-value customers (critical!)
    high_risk_valuable = master[(master['Risk_Level'] == 'High') & 
                                (master['Predicted_CLV_12m'] > 1000)]
    
    print(f"\nHigh-Risk Valuable Customers:")
    print(f"Count: {len(high_risk_valuable):,}")
    print(f"At-risk revenue: ${high_risk_valuable['Predicted_CLV_12m'].sum():,.2f}")
    
    # Save master file
    output_path = Path("data/powerbi/customers_master.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    
    logger.info(f"Master file saved to: {output_path}")
    
    # Also save a simplified version for easier Power BI import
    master_simple = master[['CustomerID', 'Segment', 'Recency', 'Frequency', 'Monetary',
                           'Predicted_CLV_12m', 'Churn_Probability', 'Risk_Level', 
                           'Recommended_Action']]
    
    simple_path = Path("data/powerbi/customers_simple.csv")
    master_simple.to_csv(simple_path, index=False)
    logger.info(f"Simplified file saved to: {simple_path}")
    
    print("\n" + "="*80)
    print("COMBINE PREDICTIONS COMPLETE")
    print("="*80)
    print(f"\nPower BI ready files:")
    print(f"  Full: {output_path}")
    print(f"  Simple: {simple_path}")
    print(f"\nNext step: Build Power BI dashboard")


if __name__ == "__main__":
    combine_all_predictions()
