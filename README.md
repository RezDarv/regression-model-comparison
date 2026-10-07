# 📈 Regression Model Comparison

A practical comparison of 8 regression algorithms — **Linear Regression, KNN,
Decision Tree, SVM (SVR), Random Forest, Gradient Boosting, XGBoost, and
LightGBM** — on the **California Housing** dataset.

The script trains every model with default parameters, benchmarks accuracy
(RMSE, R²) and speed (training/prediction time), then tunes the top 3
performers with `GridSearchCV` for a final head-to-head comparison.

---

## 📂 Project Structure

```
regression-model-comparison/
│
├── src/
│   └── regression_comparison.py   # Main script: trains, evaluates, tunes, and compares all models
│
├── results/                       # Generated plots and CSV tables are saved here
│
├── requirements.txt               # Required Python packages
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧰 Models Covered

| Model | Needs Scaling |
|---|---|
| Linear Regression | No |
| KNN | ✅ Yes |
| Decision Tree | No |
| SVM (SVR) | ✅ Yes |
| Random Forest | No |
| Gradient Boosting (sklearn) | No |
| XGBoost | No |
| LightGBM | No |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<username>/regression-model-comparison.git
cd regression-model-comparison
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the script
```bash
python src/regression_comparison.py
```

The script automatically:
1. Loads and inspects the California Housing dataset
2. Trains and evaluates all 8 models with default parameters
3. Prints a comparison table (RMSE, R², training time, prediction time)
4. Displays comparison plots (accuracy vs. training time)
5. Tunes the top 3 models using `GridSearchCV`
6. Prints a final table combining default and tuned results

---

## 📊 Sample Results

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | ≈ 0.75 | ≈ 0.58 |
| Decision Tree | ≈ 0.71 | ≈ 0.62 |
| KNN | ≈ 0.58 | ≈ 0.74 |
| SVM (SVR) | ≈ 0.57 | ≈ 0.75 |
| Random Forest | ≈ 0.50 | ≈ 0.81 |
| Gradient Boosting | ≈ 0.54 | ≈ 0.78 |
| XGBoost | ≈ 0.47 | ≈ 0.83 |
| **LightGBM** | **≈ 0.46** | **≈ 0.84** |

> Exact numbers may vary slightly depending on library versions and hardware.

---

## 🧠 Key Takeaways

- Boosting models (**XGBoost**, **LightGBM**) deliver the best accuracy on this
  dataset thanks to their ability to capture non-linear relationships and
  feature interactions.
- **Linear Regression** is the fastest and most interpretable, but underfits
  because the price/feature relationship isn't linear.
- **SVM** and **KNN** require feature scaling and don't scale well to larger
  datasets, making them less practical here than tree-based ensembles.

| If... | Use |
|---|---|
| Small data + interpretability matters | Linear Regression |
| Small data + non-linear relationship | KNN / SVM / Decision Tree |
| Medium data + high accuracy needed | Random Forest / Gradient Boosting |
| Large data + accuracy + speed | XGBoost / LightGBM |

---

## 📦 Libraries Used

- scikit-learn
- xgboost
- lightgbm
- pandas / numpy
- matplotlib

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
