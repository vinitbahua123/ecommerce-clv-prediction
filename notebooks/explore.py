"""Quick data exploration - Production style"""
import pandas as pd

# Load data
print("📥 Loading dataset...")
df = pd.read_excel("data/raw/Online_Retail.xlsx")

# Quick profile
print(f"\n📊 QUICK STATS:")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")
print(f"   Customers: {df['CustomerID'].nunique():,}")
print(f"   Products: {df['StockCode'].nunique():,}")
print(f"   Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"\n⚠️  Missing CustomerID: {df['CustomerID'].isnull().sum():,} ({df['CustomerID'].isnull().sum()/len(df)*100:.1f}%)")

# Show first few rows
print(f"\n👀 First 5 rows:")
print(df.head())

# Show columns
print(f"\n📋 Columns:")
print(df.columns.tolist())

print("\n✅ Exploration complete!")