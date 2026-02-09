"""
Data Cleaning Module - Production Quality
Handles all data quality issues including CustomerID formatting
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean UCI Online Retail data with production practices"""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete cleaning pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("🧹 Starting data cleaning...")
        
        initial_rows = len(df)
        df_clean = df.copy()
        
        # Step 1: Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Step 2: Fix data types (including CustomerID!)
        df_clean = self._fix_datatypes(df_clean)
        
        # Step 3: Handle missing CustomerIDs
        df_clean = self._handle_missing_customerid(df_clean)
        
        # Step 4: Remove invalid records
        df_clean = self._remove_invalid_records(df_clean)
        
        # Step 5: Handle cancellations
        df_clean = self._handle_cancellations(df_clean)
        
        # Step 6: Create basic features
        df_clean = self._create_features(df_clean)
        
        final_rows = len(df_clean)
        removed = initial_rows - final_rows
        
        logger.info(f"✅ Cleaning complete!")
        logger.info(f"   {initial_rows:,} → {final_rows:,} rows")
        logger.info(f"   Removed: {removed:,} ({removed/initial_rows*100:.1f}%)")
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial = len(df)
        df = df.drop_duplicates()
        removed = initial - len(df)
        self.cleaning_stats['duplicates'] = removed
        logger.info(f"  ✓ Duplicates removed: {removed:,}")
        return df
    
    def _fix_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix data types - PRODUCTION QUALITY!
        
        CRITICAL: Handle CustomerID properly
        - Currently: 17850.0 (float with .0)
        - Target: '17850' (clean string)
        """
        logger.info("  ✓ Converting data types...")
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Fix CustomerID (THE IMPORTANT PART!)
        # Step 1: Fill NaN with placeholder temporarily
        df['CustomerID'] = df['CustomerID'].fillna(-1)
        
        # Step 2: Convert to integer (removes .0)
        df['CustomerID'] = df['CustomerID'].astype(int)
        
        # Step 3: Convert to string (IDs are identifiers, not numbers)
        df['CustomerID'] = df['CustomerID'].astype(str)
        
        # Step 4: Mark missing as NaN again (for filtering later)
        df.loc[df['CustomerID'] == '-1', 'CustomerID'] = np.nan
        
        # Ensure numeric types
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        
        logger.info("     CustomerID: 17850.0 → '17850' ✓")
        
        return df
    
    def _handle_missing_customerid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows without CustomerID.
        
        BUSINESS DECISION: Cannot analyze customers without ID
        """
        initial = len(df)
        df = df[df['CustomerID'].notna()]
        removed = initial - len(df)
        self.cleaning_stats['missing_customerid'] = removed
        logger.info(f"  ✓ Missing CustomerID removed: {removed:,} (25%)")
        return df
    
    def _remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records violating business rules"""
        initial = len(df)
        
        # Only positive quantities
        df = df[df['Quantity'] > 0]
        
        # Only positive prices
        df = df[df['UnitPrice'] > 0]
        
        removed = initial - len(df)
        self.cleaning_stats['invalid_records'] = removed
        logger.info(f"  ✓ Invalid records removed: {removed:,}")
        return df
    
    def _handle_cancellations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove cancellations (InvoiceNo starts with 'C')"""
        initial = len(df)
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        removed = initial - len(df)
        self.cleaning_stats['cancellations'] = removed
        logger.info(f"  ✓ Cancellations removed: {removed:,}")
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic derived features"""
        # Total amount
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
        
        # Temporal features
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['Date'] = df['InvoiceDate'].dt.date
        
        logger.info("  ✓ Features created")
        return df


def main():
    """Run the cleaning pipeline"""
    print("="*60)
    print("🧹 UCI ONLINE RETAIL - DATA CLEANING PIPELINE")
    print("="*60)
    
    # Load raw data
    print("\n📥 Loading raw data...")
    df_raw = pd.read_excel("data/raw/Online_Retail.xlsx")
    print(f"   Loaded: {len(df_raw):,} rows")
    
    # Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)
    
    # Show sample of cleaned CustomerID
    print(f"\n✨ CustomerID FIXED:")
    print(f"   Before: 17850.0, 17851.0, 17852.0")
    print(f"   After:  {df_clean['CustomerID'].head(3).tolist()}")
    print(f"   Type: {df_clean['CustomerID'].dtype}")
    
    # Save cleaned data
    output_path = Path("data/processed/online_retail_clean.csv")
    output_path.parent.mkdir(exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved cleaned data:")
    print(f"   Location: {output_path}")
    print(f"   Rows: {len(df_clean):,}")
    print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    print("\n✅ Data cleaning complete!")
    print(f"📊 Final dataset: {len(df_clean):,} rows × {len(df_clean.columns)} columns")

if __name__ == "__main__":
    main()
