

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Production-grade data cleaning pipeline.
    
    Implements systematic data quality checks and transformations
    with comprehensive audit trail.
    """
    
    def __init__(self):
        """Initialize cleaner with audit tracking."""
        self.audit_trail = {
            'initial_rows': 0,
            'final_rows': 0,
            'steps': []
        }
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute complete data cleaning pipeline.
        
        Args:
            df: Raw DataFrame from UCI dataset
            
        Returns:
            Cleaned DataFrame
        """
        self.audit_trail['initial_rows'] = len(df)
        logger.info(f"Starting data cleaning pipeline")
        logger.info(f"Initial dataset size: {len(df):,} rows x {len(df.columns)} columns")
        
        df_clean = df.copy()
        
        # Sequential cleaning steps
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._convert_datatypes(df_clean)
        df_clean = self._handle_missing_customerid(df_clean)
        df_clean = self._remove_invalid_quantities(df_clean)
        df_clean = self._remove_invalid_prices(df_clean)
        df_clean = self._handle_cancellations(df_clean)
        df_clean = self._create_derived_features(df_clean)
        
        self.audit_trail['final_rows'] = len(df_clean)
        self._print_summary()
        
        return df_clean
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate transaction records."""
        initial = len(df)
        df = df.drop_duplicates()
        removed = initial - len(df)
        
        self.audit_trail['steps'].append({
            'step': 'Remove duplicates',
            'rows_before': initial,
            'rows_after': len(df),
            'rows_removed': removed,
            'percentage_removed': round(removed / initial * 100, 2)
        })
        
        logger.info(f"Step 1: Duplicates removed: {removed:,} rows ({removed/initial*100:.2f}%)")
        logger.info(f"        Remaining: {len(df):,} rows")
        
        return df
    
    def _convert_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate data types.
        
        Conversions:
        - InvoiceDate: object -> datetime64
        - CustomerID: float64 -> string (identifier, not numeric)
        - Quantity: ensure numeric
        - UnitPrice: ensure numeric
        """
        logger.info(f"Step 2: Converting data types")
        
        # InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        
        # CustomerID: float to string (removes .0 suffix)
        df['CustomerID'] = df['CustomerID'].fillna(-1).astype(int).astype(str)
        df.loc[df['CustomerID'] == '-1', 'CustomerID'] = np.nan
        
        # Ensure numeric types
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
        
        logger.info(f"        Data types converted successfully")
        
        return df
    
    def _handle_missing_customerid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove transactions without customer identifier.
        
        Business Rule: Customer-level analysis requires valid CustomerID.
        Cannot impute customer identity - removal is only option.
        """
        initial = len(df)
        missing_count = df['CustomerID'].isna().sum()
        
        df = df[df['CustomerID'].notna()]
        removed = initial - len(df)
        
        self.audit_trail['steps'].append({
            'step': 'Remove missing CustomerID',
            'rows_before': initial,
            'rows_after': len(df),
            'rows_removed': removed,
            'percentage_removed': round(removed / initial * 100, 2)
        })
        
        logger.info(f"Step 3: Missing CustomerID removed: {removed:,} rows ({removed/initial*100:.2f}%)")
        logger.info(f"        Remaining: {len(df):,} rows")
        logger.info(f"        Unique customers: {df['CustomerID'].nunique():,}")
        
        return df
    
    def _remove_invalid_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove transactions with invalid quantities.
        
        Business Rules:
        - Quantity must be positive (> 0)
        - Negative quantities represent returns/adjustments (exclude from CLV)
        - Zero quantities are data errors
        """
        initial = len(df)
        
        negative = (df['Quantity'] < 0).sum()
        zero = (df['Quantity'] == 0).sum()
        null = df['Quantity'].isna().sum()
        
        df = df[df['Quantity'] > 0]
        removed = initial - len(df)
        
        self.audit_trail['steps'].append({
            'step': 'Remove invalid quantities',
            'rows_before': initial,
            'rows_after': len(df),
            'rows_removed': removed,
            'percentage_removed': round(removed / initial * 100, 2),
            'detail': f'Negative: {negative:,}, Zero: {zero:,}, Null: {null:,}'
        })
        
        logger.info(f"Step 4: Invalid quantities removed: {removed:,} rows ({removed/initial*100:.2f}%)")
        logger.info(f"        Breakdown - Negative: {negative:,}, Zero: {zero:,}, Null: {null:,}")
        logger.info(f"        Remaining: {len(df):,} rows")
        
        return df
    
    def _remove_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove transactions with invalid unit prices.
        
        Business Rules:
        - UnitPrice must be positive (> 0)
        - Free items (price = 0) excluded as data anomalies
        - Negative prices are data errors
        """
        initial = len(df)
        
        negative = (df['UnitPrice'] < 0).sum()
        zero = (df['UnitPrice'] == 0).sum()
        null = df['UnitPrice'].isna().sum()
        
        df = df[df['UnitPrice'] > 0]
        removed = initial - len(df)
        
        self.audit_trail['steps'].append({
            'step': 'Remove invalid prices',
            'rows_before': initial,
            'rows_after': len(df),
            'rows_removed': removed,
            'percentage_removed': round(removed / initial * 100, 2),
            'detail': f'Negative: {negative:,}, Zero: {zero:,}, Null: {null:,}'
        })
        
        logger.info(f"Step 5: Invalid prices removed: {removed:,} rows ({removed/initial*100:.2f}%)")
        logger.info(f"        Breakdown - Negative: {negative:,}, Zero: {zero:,}, Null: {null:,}")
        logger.info(f"        Remaining: {len(df):,} rows")
        
        return df
    
    def _handle_cancellations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process cancellation transactions.
        
        Business Rule: InvoiceNo starting with 'C' indicates cancellation.
        Excluded from CLV analysis (focus on positive purchases).
        """
        initial = len(df)
        
        cancellation_count = df['InvoiceNo'].astype(str).str.startswith('C').sum()
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        removed = initial - len(df)
        
        self.audit_trail['steps'].append({
            'step': 'Remove cancellations',
            'rows_before': initial,
            'rows_after': len(df),
            'rows_removed': removed,
            'percentage_removed': round(removed / initial * 100, 2)
        })
        
        logger.info(f"Step 6: Cancellations removed: {removed:,} rows ({removed/initial*100:.2f}%)")
        logger.info(f"        Remaining: {len(df):,} rows")
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for analysis.
        
        Features created:
        - TotalAmount: Quantity * UnitPrice
        - Temporal: Year, Month, Quarter, DayOfWeek, Hour
        - Date: Simplified date field
        """
        logger.info(f"Step 7: Creating derived features")
        
        # Revenue per line item
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
        
        # Temporal features
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['Date'] = df['InvoiceDate'].dt.date
        
        logger.info(f"        Features created: TotalAmount, Year, Month, Quarter, DayOfWeek, Hour, Date")
        logger.info(f"        Final dataset: {len(df):,} rows x {len(df.columns)} columns")
        
        return df
    
    def _print_summary(self):
        """Print comprehensive cleaning summary."""
        print("\n" + "="*80)
        print("DATA CLEANING AUDIT REPORT")
        print("="*80)
        
        print(f"\nInitial dataset: {self.audit_trail['initial_rows']:,} rows")
        print(f"Final dataset: {self.audit_trail['final_rows']:,} rows")
        
        total_removed = self.audit_trail['initial_rows'] - self.audit_trail['final_rows']
        retention_rate = (self.audit_trail['final_rows'] / self.audit_trail['initial_rows']) * 100
        
        print(f"\nTotal removed: {total_removed:,} rows ({total_removed/self.audit_trail['initial_rows']*100:.2f}%)")
        print(f"Retention rate: {retention_rate:.2f}%")
        
        print("\n" + "-"*80)
        print("STEP-BY-STEP BREAKDOWN")
        print("-"*80)
        
        for i, step in enumerate(self.audit_trail['steps'], 1):
            print(f"\n{i}. {step['step']}")
            print(f"   Rows before: {step['rows_before']:,}")
            print(f"   Rows removed: {step['rows_removed']:,}")
            print(f"   Rows after: {step['rows_after']:,}")
            print(f"   Percentage removed: {step['percentage_removed']:.2f}%")
            if 'detail' in step:
                print(f"   Detail: {step['detail']}")
        
        print("\n" + "="*80)
        print(f"FINAL DATASET STATISTICS")
        print("="*80)
    
    def get_audit_trail(self) -> Dict:
        """Return complete audit trail for documentation."""
        return self.audit_trail


class DataValidator:
    """
    Validate cleaned data meets quality standards.
    """
    
    @staticmethod
    def validate(df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Execute comprehensive data quality validation.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Tuple of (validation_passed, list_of_failures)
        """
        logger.info("Executing data quality validation")
        
        failures = []
        
        # Check 1: No null CustomerIDs
        null_customers = df['CustomerID'].isna().sum()
        if null_customers > 0:
            failures.append(f"Found {null_customers:,} null CustomerIDs")
        
        # Check 2: All quantities positive
        invalid_qty = (df['Quantity'] <= 0).sum()
        if invalid_qty > 0:
            failures.append(f"Found {invalid_qty:,} non-positive quantities")
        
        # Check 3: All prices positive
        invalid_price = (df['UnitPrice'] <= 0).sum()
        if invalid_price > 0:
            failures.append(f"Found {invalid_price:,} non-positive prices")
        
        # Check 4: Valid dates
        invalid_dates = df['InvoiceDate'].isna().sum()
        if invalid_dates > 0:
            failures.append(f"Found {invalid_dates:,} invalid dates")
        
        # Check 5: No duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            failures.append(f"Found {duplicates:,} duplicate rows")
        
        # Check 6: Required columns present
        required_columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                          'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            failures.append(f"Missing required columns: {missing_columns}")
        
        # Check 7: Minimum data volume
        if len(df) < 100000:
            failures.append(f"Warning: Dataset contains only {len(df):,} rows (expected >300,000)")
        
        validation_passed = len(failures) == 0
        
        if validation_passed:
            logger.info("Data validation PASSED - All quality checks successful")
        else:
            logger.warning(f"Data validation FAILED - {len(failures)} issues found")
            for failure in failures:
                logger.warning(f"  - {failure}")
        
        return validation_passed, failures


def main():
    """Execute data cleaning pipeline."""
    
    print("="*80)
    print("UCI ONLINE RETAIL DATASET - DATA CLEANING PIPELINE")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    input_path = Path("data/raw/Online_Retail.xlsx")
    output_path = Path("data/processed/online_retail_clean.csv")
    
    # Validate input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error(f"Please run download_data.py first")
        return
    
    # Load raw data
    logger.info(f"Loading raw data from: {input_path}")
    df_raw = pd.read_excel(input_path, engine='openpyxl')
    logger.info(f"Raw data loaded: {len(df_raw):,} rows x {len(df_raw.columns)} columns")
    
    # Execute cleaning
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)
    
    # Validate cleaned data
    validator = DataValidator()
    is_valid, failures = validator.validate(df_clean)
    
    if not is_valid:
        print("\nWARNING: Data validation failed")
        print("Issues found:")
        for failure in failures:
            print(f"  - {failure}")
        print("\nReview cleaning logic before proceeding")
        return
    
    # Additional statistics
    print("\nCLEANED DATASET CHARACTERISTICS")
    print("-"*80)
    print(f"Unique customers: {df_clean['CustomerID'].nunique():,}")
    print(f"Unique products: {df_clean['StockCode'].nunique():,}")
    print(f"Unique invoices: {df_clean['InvoiceNo'].nunique():,}")
    print(f"Countries: {df_clean['Country'].nunique()}")
    print(f"Date range: {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")
    print(f"Total revenue: ${df_clean['TotalAmount'].sum():,.2f}")
    
    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Cleaned data saved to: {output_path}")
    logger.info(f"File size: {file_size_mb:.2f} MB")
    
    # Save audit trail
    audit_path = output_path.parent / "cleaning_audit_trail.json"
    import json
    with open(audit_path, 'w') as f:
        json.dump(cleaner.get_audit_trail(), f, indent=2, default=str)
    logger.info(f"Audit trail saved to: {audit_path}")
    
    print("\n" + "="*80)
    print("DATA CLEANING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
