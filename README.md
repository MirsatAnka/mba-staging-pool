# Market Basket Analysis on Online Retail II (Topic 6)

### Group 6 | SEED = 5018

## Group Members
- **Mirsat Anka** (221015018)
- **Mustafa Adalan** (221015025)
- **Emir Ürü** (221015035)
- **Zeynep Tek** (221015062)
- **Çağatay Utku Çamyar** (241015407)

**Instructor Collaborator:** `@AkbulutOsman`  


---

## Project Overview
This repository implements an end-to-end **Market Basket Analysis (Association Rule Mining)** pipeline on the **UCI Online Retail II** dataset (ID: 502, ~1.06M transaction records covering UK-based online retail activity between December 2009 and December 2011).

To preserve market basket integrity on >1M line items, **random row-level sampling is strictly prohibited**. Instead, our pipeline filters out cancelled invoices (`InvoiceNo` prefixed with `'C'`), cleans invalid quantities and unit prices, removes missing product descriptions/customer identifiers, and performs **deterministic invoice-level sampling** (`random_state=5018`) before constructing a sparse boolean invoice-by-item basket matrix for frequent itemset mining via `mlxtend`.

---

## Research Questions (RQs)
1. **RQ1 (Threshold Sensitivity & Rule Quality):** How do varying minimum support and confidence thresholds affect the number of association rules generated and their respective lift and conviction distributions?
2. **RQ2 (Algorithmic Efficiency):** How do Apriori and FP-Growth algorithms compare in terms of execution time, memory usage, and rule generation efficiency on the sampled invoice transactions?
3. **RQ3 (Actionable Cross-Selling Insights):** Which product categories exhibit the strongest cross-selling associations (highest lift), and how can these rules be translated into actionable retail layout strategies?

---

## Execution Guide

### Prerequisites
- **Python 3.10+**
- Standard laptop CPU (full pipeline completes well under 30 minutes)

### Step-by-Step Reproduction
```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Download and extract the UCI Online Retail II dataset into data/
python src/download_data.py

# 4. Run the complete preprocessing, sampling (SEED=5018), and benchmarking pipeline
python run_all.py
```

---

## Folder Structure
```text
.
├── .gitignore
├── README.md
├── JOURNAL.md
├── requirements.txt
├── run_all.py
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── preprocessing.py
│   └── association_rules.py
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   └── tables/
│       └── .gitkeep
└── report/
    └── .gitkeep
```

---

## Main Results Summary (Placeholder)
*Generated automatically into `results/tables/algorithm_benchmark.csv` upon executing `python run_all.py`.*

| Algorithm | Min Support | Min Confidence | Frequent Itemsets | Rules Generated | Avg Lift | Avg Conviction | Execution Time (s) | Peak Memory (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Apriori | 0.030 | 0.50 | TBD | TBD | TBD | TBD | TBD | TBD |
| FP-Growth | 0.030 | 0.50 | TBD | TBD | TBD | TBD | TBD | TBD |
| Apriori | 0.020 | 0.50 | TBD | TBD | TBD | TBD | TBD | TBD |
| FP-Growth | 0.020 | 0.50 | TBD | TBD | TBD | TBD | TBD | TBD |
| Apriori | 0.015 | 0.50 | TBD | TBD | TBD | TBD | TBD | TBD |
| FP-Growth | 0.015 | 0.50 | TBD | TBD | TBD | TBD | TBD | TBD |
| Apriori | 0.010 | 0.50 | TBD | TBD | TBD | TBD | TBD | TBD |
| FP-Growth | 0.010 | 0.50 | TBD | TBD | TBD | TBD | TBD | TBD |

---

## Dataset License & Citation
- **Dataset:** Online Retail II (UCI Machine Learning Repository, Dataset ID: 502)
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)
- **Citation:**
  > Chen, Daqing. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
