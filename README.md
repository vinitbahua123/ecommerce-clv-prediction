# E-Commerce Customer Lifetime Value Prediction

A comprehensive customer analytics platform leveraging machine learning to segment customers, predict lifetime value, and identify churn risks for data-driven marketing optimization.

---

## Project Overview

This project implements an end-to-end customer analytics system using the UCI Online Retail dataset (541,909 transactions). The system segments customers into actionable tiers, forecasts 12-month customer lifetime value, and identifies churn risks to optimize marketing spend and retention strategies.

### Key Results

- **Customer Segmentation:** 4,338 customers grouped into 4 distinct tiers
- **Revenue Concentration:** Top 5% of customers generate 48.5% of revenue
- **CLV Prediction:** $8.6M total forecasted revenue over 12 months
- **Churn Detection:** 234 high-risk customers identified (ROC-AUC: 0.876)
- **Model Performance:** K-Means Silhouette 0.617, CLV models trained, Churn AUC 0.876

---

## Technical Stack

**Programming Language:** Python 3.11+  
**Data Processing:** pandas, numpy  
**Machine Learning:** scikit-learn, XGBoost, LightGBM, lifetimes  
**Visualization:** matplotlib, seaborn, plotly  
**Database:** PostgreSQL (optional)  
**Dashboard:** Power BI  

---

## Project Structure
```
ecommerce-clv-prediction/
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Cleaned data
│   ├── features/                 # RFM metrics, segments
│   └── predictions/              # CLV and churn predictions
├── src/
│   ├── data/                     # Data processing modules
│   ├── models/                   # ML model implementations
│   ├── utils/                    # Utility functions
│   └── visualizations/           # Chart generation
├── models/                       # Trained model files
├── docs/                         # Documentation
├── notebooks/                    # Jupyter notebooks
└── requirements.txt              # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions
```bash
# Clone repository
git clone https://github.com/yourusername/ecommerce-clv-prediction.git
cd ecommerce-clv-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download dataset
python3 download_data.py
```

---

## Usage

### Complete Pipeline Execution

Run the following scripts in sequence:
```bash
# Step 1: Data Cleaning
python3 src/data/clean_data.py
# Output: data/processed/online_retail_clean.csv (392,692 transactions)

# Step 2: RFM Calculation
python3 src/data/feature_engineering.py
# Output: data/features/customer_rfm.csv (4,338 customers)

# Step 3: Customer Segmentation
python3 src/models/rfm_segmentation.py
# Output: data/features/customer_segments.csv

# Step 4: CLV Prediction
python3 src/models/clv_prediction.py
# Output: data/predictions/customer_clv.csv

# Step 5: Churn Detection
python3 src/models/churn_classifier.py
# Output: data/predictions/customer_churn.csv

# Step 6: Combine Predictions
python3 src/utils/combine_predictions.py
# Output: data/powerbi/customers_master.csv

# Step 7: Generate Visualizations
python3 src/visualizations/segment_charts.py
# Output: docs/figures/*.png
```

---

## Model Descriptions

### Customer Segmentation (K-Means)

**Algorithm:** K-Means clustering  
**Features:** Recency, Frequency, Monetary (RFM)  
**Output:** 4 customer segments  
**Performance:** Silhouette Score 0.617

**Segments:**
- **Platinum Tier:** 14 customers (0.3%) - $113K average value, 18% revenue
- **Gold Tier:** 213 customers (4.9%) - $12.8K average value, 31% revenue
- **Silver Tier:** 3,048 customers (70.3%) - $1.3K average value, 46% revenue
- **Inactive:** 1,063 customers (24.5%) - $479 average value, 6% revenue

---

### CLV Prediction (BG/NBD + Gamma-Gamma)

**Algorithms:** Beta-Geometric/NBD and Gamma-Gamma models  
**Features:** Customer frequency, recency, and monetary metrics  
**Prediction Horizon:** 12 months  
**Output:** Revenue forecast per customer

**Results:**
- Total predicted revenue: $8,622,048
- Average CLV: $1,988
- Median CLV: $1,022
- Top customer: $222,167 predicted value

---

### Churn Classification (LightGBM)

**Algorithm:** LightGBM Gradient Boosting  
**Features:** 12 behavioral and RFM-derived features  
**Churn Definition:** No purchase in 90+ days  
**Class Balance:** SMOTE over-sampling

**Performance:**
- ROC-AUC: 0.876
- Precision: 0.92
- Recall: 0.76
- High-risk customers identified: 234

---

## Key Findings

### Business Insights

1. **Revenue Concentration:** Top 5.2% of customers (227 total) generate 48.5% of total revenue
2. **Pareto Principle:** Top 20% of customers contribute 63.3% of predicted future revenue
3. **Churn Risk:** 234 high-risk customers represent approximately $445K in at-risk revenue
4. **Customer Tiers:** Clear segmentation enables targeted marketing strategies

### Model Performance

- **Segmentation Quality:** Silhouette score 0.617 indicates well-separated clusters
- **CLV Accuracy:** Statistical models provide reliable 12-month forecasts
- **Churn Detection:** 87.6% AUC demonstrates excellent predictive capability

---

## Output Files

### Primary Outputs

**customers_master.csv**  
Combined dataset with all predictions (4,338 rows × 16 columns)

**Columns:**
- Customer identification and RFM metrics
- Segment classification
- 12-month CLV prediction
- Churn probability and risk level
- Recommended business actions

**segment_profiles.csv**  
Aggregate statistics for each customer segment (4 rows)

---

## Requirements

### Python Dependencies
```
pandas>=2.1.0
numpy>=1.25.0
scikit-learn>=1.3.0
lightgbm>=4.1.0
lifetimes>=0.11.3
matplotlib>=3.8.0
seaborn>=0.12.2
plotly>=5.17.0
imbalanced-learn>=0.11.0
dill>=0.3.7
```

Full dependencies listed in `requirements.txt`

---

## Data

### Source

**Dataset:** UCI Online Retail Dataset  
**URL:** https://archive.ics.uci.edu/dataset/352/online+retail  
**License:** CC BY 4.0  
**Citation:** Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.

### Dataset Statistics

**Raw Data:**
- Transactions: 541,909
- Time Period: December 2010 - December 2011
- Customers: 4,372
- Products: 4,070
- Countries: 38

**Clean Data:**
- Transactions: 392,692 (72.5% retention)
- Customers: 4,338
- Data Quality Score: 95%+

---

## Methodology

### Data Preparation

1. **Data Cleaning:** Remove duplicates, handle missing values, validate business rules
2. **Feature Engineering:** Calculate RFM metrics, create derived features
3. **Data Validation:** Quality checks and audit trail generation

### Machine Learning Pipeline

1. **Segmentation:** K-Means clustering on scaled RFM features
2. **CLV Prediction:** BG/NBD model for frequency, Gamma-Gamma for monetary value
3. **Churn Detection:** LightGBM classifier with SMOTE for class balance

### Evaluation

- Segmentation: Silhouette score, Davies-Bouldin index
- CLV: Statistical model convergence, business logic validation
- Churn: ROC-AUC, precision-recall analysis, confusion matrix

---

## Team

**Project Lead:** Shrividya Shashidhara  
**Data Analyst:** Praniti Kale  
**ML Engineer:** Vinit Bahua  
**Data Engineer:** Diego Lacruz

**Institution:** Northeastern University  
**Program:** Data Science Graduate Program

---

## Documentation

- **Data Dictionary:** `docs/DATA_DICTIONARY_TEAM_REFERENCE.md`
- **Project Roadmap:** `docs/PROJECT_ROADMAP_SIMPLE.md`
- **Requirements Flow:** `docs/REQUIREMENTS_FLOW_DOCUMENT.md`

---

## Deliverables

1. **Data Storage:** PostgreSQL database with cleaned transaction data
2. **ML Models:** K-Means segmentation, CLV prediction, churn classifier
3. **Data Pipeline:** Automated ETL workflow with validation
4. **Dashboard:** Interactive Power BI application (5 pages)
5. **Repository:** Complete codebase with documentation

---

## Future Enhancements

- Real-time streaming data integration
- A/B testing framework for retention strategies
- Product recommendation engine
- Multi-channel campaign automation
- Advanced customer journey mapping

---

## License

MIT License - See LICENSE file for details

---

## Contact

**Repository:** github.com/yourusername/ecommerce-clv-prediction  
**Issues:** Use GitHub Issues for bug reports or feature requests

---

## Acknowledgments

Dataset provided by UCI Machine Learning Repository. This project was developed as part of the Data Science curriculum at Northeastern University.

---

**Last Updated:** February 2026  
**Status:** Production Ready
