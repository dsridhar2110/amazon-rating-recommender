# FIT5212 Assignment 2 — Recommender System Challenge

A hybrid recommender system for the [FIT5212 S1 2025 Kaggle competition](https://www.kaggle.com/competitions/fit-5212-s-1-2025). The system predicts user–item ratings (1–5) on Amazon-style product review data by combining **matrix factorization (SVD)**, **content-based filtering (TF-IDF)**, and **bias/fallback** strategies.

## Approach

- **Collaborative filtering:** Surprise library SVD on user–product interactions (for products with sufficient ratings).
- **Content-based:** TF-IDF on product names to handle cold-start and rarely rated items.
- **Fallbacks:** Global mean and user/product bias when neither SVD nor similar products are available.
- **Optional:** Quality/helpfulness-based adjustments for high-confidence predictions.

The hybrid approach clearly outperforms all alternatives while maintaining reasonable computational complexity.

### 6.3 Hybrid system performance

**Case-wise prediction analysis:**

- **Enhanced SVD (Case 1):** 85.1% of predictions — handles products with rich collaborative data.
- **Statistical prediction (Case 2):** 7.0% of predictions — manages products with limited data.
- **Content similarity (Case 3):** 7.9% of predictions — addresses completely new products.

This distribution aligns well with EDA findings: the majority of cases benefit from the SVD approach while fallback methods handle edge cases effectively.

**Coverage analysis:** The system provides predictions for **100%** of test cases with no fallback to simple global mean, indicating robust handling of all scenarios including cold-start.

### 6.4 Error analysis

**Prediction distribution:**

- **Average prediction:** 4.237 (close to training mean of 4.239).
- **Standard deviation:** 0.749 (reasonable spread).
- **Range:** 1.000 to 5.000 (properly bounded).
- **Low ratings (<3.0):** 6.4% of predictions.
- **High ratings (>4.5):** 39.9% of predictions.

The prediction distribution closely matches the training data characteristics, suggesting good model calibration without systematic bias.

**Error patterns:** Most prediction errors occur in the middle rating ranges (3–4) where user preferences are most ambiguous. The model shows good performance on extreme ratings (1–2, 5) where user sentiment is clearer.

## Repository structure

**Current (flat — recommended for this size):**

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── code.ipynb              # Main notebook (EDA, models, training, submission)
├── Final_predictions.csv   # Best submission (ID, rating) for Kaggle/Moodle
├── report.pdf              # Discussion report (optional; add when ready)
├── train.csv               # Training data (gitignored if large; get from Kaggle)
└── test.csv                # Test data (gitignored if large; get from Kaggle)
```

You can optionally put `train.csv` and `test.csv` in a **`data/`** folder and set `DATA_DIR = "data"` in the path-setup cell; the rest can stay at the root. No need to move the notebook or submission into folders for this project.

## Setup

1. **Clone the repo** (or download the files).

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get the data** from the competition:
   - [FIT5212 S1 2025 — Kaggle](https://www.kaggle.com/competitions/fit-5212-s-1-2025/data)
   - Download `train.csv` and `test.csv` and place them in this folder (or in a `data/` subfolder and set paths in the notebook).

## How to run

1. Open `code.ipynb` in Jupyter or VS Code.
2. Run the **path setup** cell near the top: it sets `TRAIN_PATH` and `TEST_PATH` automatically (Kaggle vs local). For local runs, put `train.csv` and `test.csv` in the project folder, or set `DATA_DIR` in that cell (e.g. to `"data"`).
3. Run all cells. The main pipeline is in **Step 4 (Hybrid Recommender)**; the last execution calls `main()`, which:
   - Trains on a user-based train/validation split,
   - Validates (RMSE, R², MAE),
   - Retrains on the full training set,
   - Writes predictions to a submission CSV (e.g. `hybrid_submission.csv`; rename/copy to `Final_predictions.csv` for Moodle if needed).
4. For Moodle/Kaggle: use `Final_predictions.csv` (or your best submission) with exactly two columns — **ID** and **rating**.

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions.

## License

For academic use (FIT5212 Monash University). Do not copy for assignment submission; this is individual work.
