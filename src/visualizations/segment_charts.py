"""
Customer Segmentation Visualizations
Generate professional charts for segment analysis


Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set professional styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_3d_scatter_plot(df: pd.DataFrame, output_dir: Path):
    """
    Create 3D scatter plot of RFM segmentation.
    
    Args:
        df: DataFrame with Recency, Frequency, Monetary, Segment
        output_dir: Directory to save figure
    """
    logger.info("Creating 3D RFM scatter plot")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each segment with different color
    segments = df['Segment'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, segment in enumerate(sorted(segments)):
        segment_data = df[df['Segment'] == segment]
        ax.scatter(
            segment_data['Recency'],
            segment_data['Frequency'],
            segment_data['Monetary'],
            c=colors[i % len(colors)],
            label=segment,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('Recency (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (orders)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Monetary ($)', fontsize=12, fontweight='bold')
    ax.set_title('Customer Segmentation - 3D RFM Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'segment_3d_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def create_segment_distribution(profiles: pd.DataFrame, output_dir: Path):
    """
    Create segment distribution charts (customer count and revenue).
    
    Args:
        profiles: Segment profiles DataFrame
        output_dir: Directory to save figure
    """
    logger.info("Creating segment distribution charts")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Customer Count Distribution
    axes[0].barh(profiles['Segment'], profiles['Customer_Count'], color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Number of Customers', fontsize=12, fontweight='bold')
    axes[0].set_title('Customer Distribution by Segment', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    for i, (count, pct) in enumerate(zip(profiles['Customer_Count'], profiles['Customer_Percentage'])):
        axes[0].text(count, i, f' {count:,} ({pct:.1f}%)', 
                    va='center', fontsize=10, fontweight='bold')
    
    # Chart 2: Revenue Contribution
    axes[1].barh(profiles['Segment'], profiles['Revenue_Percentage'], color='coral', edgecolor='black')
    axes[1].set_xlabel('Revenue Contribution (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Revenue Contribution by Segment', fontsize=14, fontweight='bold')
    
    # Add labels
    for i, (rev_pct, total_rev) in enumerate(zip(profiles['Revenue_Percentage'], profiles['Total_Revenue'])):
        axes[1].text(rev_pct, i, f' {rev_pct:.1f}% (${total_rev:,.0f})', 
                    va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'segment_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def create_segment_comparison_heatmap(profiles: pd.DataFrame, output_dir: Path):
    """
    Create heatmap comparing segment characteristics.
    
    Args:
        profiles: Segment profiles DataFrame
        output_dir: Directory to save figure
    """
    logger.info("Creating segment comparison heatmap")
    
    # Prepare data for heatmap
    heatmap_data = profiles[['Segment', 'Avg_Recency_Days', 'Avg_Frequency', 'Avg_Monetary_Value']].copy()
    heatmap_data = heatmap_data.set_index('Segment')
    
    # Normalize for better visualization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        columns=heatmap_data.columns,
        index=heatmap_data.index
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        heatmap_normalized.T,
        annot=heatmap_data.T.values,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Normalized Value'},
        linewidths=2,
        linecolor='white',
        ax=ax
    )
    
    ax.set_title('Segment Characteristics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Customer Segment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'segment_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def create_revenue_concentration_chart(df: pd.DataFrame, output_dir: Path):
    """
    Create cumulative revenue concentration chart.
    
    Args:
        df: Customer segments DataFrame
        output_dir: Directory to save figure
    """
    logger.info("Creating revenue concentration chart")
    
    # Sort by monetary value
    df_sorted = df.sort_values('Monetary', ascending=False).reset_index(drop=True)
    df_sorted['Customer_Rank'] = range(1, len(df_sorted) + 1)
    df_sorted['Cumulative_Revenue'] = df_sorted['Monetary'].cumsum()
    df_sorted['Cumulative_Revenue_Pct'] = (df_sorted['Cumulative_Revenue'] / df_sorted['Monetary'].sum()) * 100
    df_sorted['Customer_Pct'] = (df_sorted['Customer_Rank'] / len(df_sorted)) * 100
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(df_sorted['Customer_Pct'], df_sorted['Cumulative_Revenue_Pct'], 
            linewidth=3, color='darkblue', label='Actual Distribution')
    
    # Add 80/20 reference line
    ax.plot([0, 100], [0, 100], '--', linewidth=2, color='gray', alpha=0.5, label='Equal Distribution')
    ax.axvline(x=20, color='red', linestyle='--', linewidth=2, alpha=0.7, label='20% Mark')
    
    # Highlight 20% point
    revenue_at_20 = df_sorted[df_sorted['Customer_Pct'] <= 20]['Cumulative_Revenue_Pct'].iloc[-1]
    ax.axhline(y=revenue_at_20, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(22, revenue_at_20, f'{revenue_at_20:.1f}% of revenue', 
            fontsize=12, fontweight='bold', color='red')
    
    ax.set_xlabel('Cumulative % of Customers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative % of Revenue', fontsize=12, fontweight='bold')
    ax.set_title('Revenue Concentration - Pareto Analysis', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'revenue_concentration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def create_segment_box_plots(df: pd.DataFrame, output_dir: Path):
    """
    Create box plots showing RFM distribution by segment.
    
    Args:
        df: Customer segments DataFrame
        output_dir: Directory to save figure
    """
    logger.info("Creating segment box plots")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Recency
    sns.boxplot(data=df, x='Segment', y='Recency', ax=axes[0], palette='Set2')
    axes[0].set_title('Recency by Segment', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Days Since Last Purchase', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Segment', fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Frequency
    sns.boxplot(data=df, x='Segment', y='Frequency', ax=axes[1], palette='Set2')
    axes[1].set_title('Frequency by Segment', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Number of Orders', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Segment', fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Monetary (log scale for better visualization)
    sns.boxplot(data=df, x='Segment', y='Monetary', ax=axes[2], palette='Set2')
    axes[2].set_title('Monetary by Segment', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Total Spend ($)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Segment', fontsize=11, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = output_dir / 'segment_boxplots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all segmentation visualizations."""
    
    print("="*80)
    print("CUSTOMER SEGMENTATION VISUALIZATIONS")
    print("="*80)
    print()
    
    # Create output directory
    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading segmentation data")
    segments_df = pd.read_csv("data/features/customer_segments.csv")
    profiles_df = pd.read_csv("data/features/segment_profiles.csv")
    
    logger.info(f"Loaded: {len(segments_df):,} customers, {len(profiles_df)} segments")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    print("-"*80)
    
    create_3d_scatter_plot(segments_df, output_dir)
    create_segment_distribution(profiles_df, output_dir)
    create_segment_comparison_heatmap(profiles_df, output_dir)
    create_revenue_concentration_chart(segments_df, output_dir)
    create_segment_box_plots(segments_df, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nGenerated 5 charts in: {output_dir}/")
    print("\nFiles created:")
    print("  1. segment_3d_scatter.png - 3D visualization of RFM clusters")
    print("  2. segment_distribution.png - Customer and revenue distribution")
    print("  3. segment_heatmap.png - Segment characteristics comparison")
    print("  4. revenue_concentration.png - Pareto analysis")
    print("  5. segment_boxplots.png - RFM distribution by segment")
    print("\nOpen these images to view your segmentation results!")


if __name__ == "__main__":
    main()
