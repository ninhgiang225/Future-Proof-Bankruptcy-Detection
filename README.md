# FutureProof: Bankruptcy Detection

A machine learning project that analyzes and predicts company bankruptcies using financial data from the Taiwan Economic Journal (1999–2009). Bankruptcy labels are defined in accordance with Taiwan Stock Exchange regulations.

---

## 🛠 Skills & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02A88E?style=for-the-badge&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-6A0DAD?style=for-the-badge&logoColor=white)
![SVM](https://img.shields.io/badge/SVM-E34F26?style=for-the-badge&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random%20Forest-228B22?style=for-the-badge&logoColor=white)

---

## Getting Started

1. **Explore the analysis** — Open `CompanyBankruptcyPrediction.ipynb` to review the full findings, methodology, and key insights. A dashboard preview is included at the end of the notebook.

2. **Launch the dashboard** — Run the following command in your terminal:

   ```bash

    # 1. Clone the repository
    
    git clone https://github.com/ninhgiang225/Future-Proof-Bankruptcy-Detection.git
    cd Future-Proof-Bankruptcy-Detection
    
    # 2. Create virtual environment (recommended)
    
    python -m venv venv
    source venv/bin/activate      # Mac/Linux
    venv\Scripts\activate         # Windows
    
    # 3. Install system dependency (macOS only — required by LightGBM)

    brew install libomp

    # 4. Install Python dependencies

    pip install -r requirements.txt
    
    # 5. Run the application
    
    streamlit run application.py
   
   ```

   This will open an interactive web app locally. Enter a company's financial metrics to receive a bankruptcy risk prediction, along with guidance on which financial indicators to monitor and improve.

<img width="939" height="463.5" alt="image" src="https://github.com/user-attachments/assets/7795d6f7-6a9d-46cd-bc49-fcf5aec33adb" />
<img width="939" height="463.5" alt="image" src="https://github.com/user-attachments/assets/7f12ac02-c20d-4adb-8871-b8368a4610fc" />


---

## Dataset Overview

| Attribute | Detail |
|---|---|
| Source | Taiwan Economic Journal |
| Period | 1999–2009 |
| Companies | 6,819 |
| Non-bankrupt | 6,599 (96.8%) |
| Bankrupt | 220 (3.2%) |

Key features include financial ratios such as ROA, operating profit rate, borrowing dependency, and equity-to-liability ratio. Class imbalance was addressed using **SMOTE** (Synthetic Minority Oversampling Technique).

---

## Modeling

**Preprocessing:** The dataset required no imputation (no missing values or duplicates) and no scaling, as values were already normalized.

**Models evaluated:** Logistic Regression, Neural Network, XGBoost, Random Forest, LightGBM, SVM, and others.

**Best model performance:**

| Metric | Score |
|---|---|
| Accuracy | 94% |
| Precision (Bankruptcy) | 98% |
| Recall (Bankruptcy) | 100% |
| F1-Score | 99% |

---

## Key Predictors of Bankruptcy

1. **Return on Assets (ROA)** — Lower ROA is strongly associated with bankruptcy risk.
2. **Operating Profit Rate** — Sustained negative values indicate deteriorating financial health.
3. **Borrowing Dependency** — High reliance on debt is a significant risk signal.
4. **Equity-to-Liability Ratio** — Lower ratios reflect elevated financial vulnerability.
5. **Net Income to Stockholders' Equity** — Persistently low returns suggest operational inefficiency.

---

## Limitations

- **Class imbalance** — Despite SMOTE, residual imbalance may affect precision on real-world data.
- **Feature collinearity** — Some financial ratios may overlap, introducing redundancy.
- **Temporal scope** — Results are specific to Taiwan's economic context from 1999–2009 and may not generalize to other periods or markets.
- **Data quality** — Model performance depends on the accuracy of the source data.

---

## Potential Improvements

- Combine SMOTE with under-sampling or ensemble resampling strategies.
- Engineer derived features such as lagged financial indicators.
- Explore deep learning architectures for capturing complex non-linear relationships.
- Adjust financial ratios to account for inflation or macroeconomic shifts.

---

## Conclusion

This project demonstrates the viability of machine learning for early bankruptcy detection using financial ratios. By tackling class imbalance and identifying the most predictive indicators, the model offers actionable insights for risk assessment. Future iterations should explore broader datasets, updated economic conditions, and more sophisticated modeling approaches.
