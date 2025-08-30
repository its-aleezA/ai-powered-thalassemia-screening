# 🩺 AI-Powered Thalassemia Screening

A machine learning model designed to detect thalassemia from blood test results, **prioritizing clinical safety by minimizing false positives**. Optimized with XGBoost and validated for real-world utility.

> **Note on Data Confidentiality**: The patient dataset used to develop this model is confidential and not publicly available due to privacy restrictions. This repository contains the source code, methodology, and analysis results only.

**Key Achievements**:
- ✅ **4.2% False Positive Rate** (exceeds clinical safety target)
- ✅ **81% Sensitivity** (detects true thalassemia cases)
- ✅ **Fully Interpretable** with SHAP explainability
- ✅ **Tiered Output** (Immediate Treatment, Urgent Testing, Follow-up)

---

## 📖 Overview

This project develops a clinical decision support tool that analyzes standard blood electrophoresis features (RBC, MCV, MCH, etc.) to screen for thalassemia. Unlike a standard classifier, it is explicitly optimized to avoid misdiagnosing healthy patients, making it safer for potential clinical use.

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
pip install -r requirements.txt
```
### 2. Basic Usage
```python
from src.modeling import ThalassemiaScreener

# Load your data (array of patient features)
X, y = load_data(...)

# Initialize and train the screener
screener = ThalassemiaScreener()
screener.fit(X, y)

# Predict on new data
patient_data = [...]
decision = screener.predict(patient_data)
```
### 3. Run the Full Analysis
Clone the repo and run the Jupyter notebooks in order:
```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

---

## 📊 Results Summary

| Metric | Normal Class | Thalassemia Class | Overall |
|---------|-----------|---------|------------|
|Precision | 0.96	| 0.65 | 0.91 |
|Recall | 0.88 | 0.81 | 0.89 |
|F1-Score | 0.92 | 0.72 | 0.90 |

ROC-AUC: 0.916 | Optimal Threshold: 0.251

---

## 🗂️ Project Structure

```text
├── notebook/       # End-to-end analysis
├── docs/           # Documentations
└── results/        # Key figures & outputs
```

---

## 🔍 Further Reading

For a deep dive into the methodology, analysis, and clinical implications, see the full project report:
-   **[📄 Comprehensive Project Report (Web Version)](docs/Thalassaemia-Model-Report.md)**
-   **[📝 Download Report as PDF](docs/Thalassaemia-Model-Report.pdf)**

---

## 👤 Author

[Aleeza Rizwan](https://github.com/its-aleezA) | Research Intern @ BRISC
Supervisor: Dr. Usman Akram

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


_Disclaimer: For research purposes only. Not intended for direct clinical use._
