# Evaluating Algorithmic Bias in Clinical Decision Systems 🩺⚖️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Fairlearn](https://img.shields.io/badge/Audited_with-Fairlearn-5c3ee8.svg)](https://fairlearn.org/)

## Overview
This repository contains the code and research artifacts for auditing algorithmic bias within clinical machine learning models. Motivated by the White House Office of Science and Technology Policy (OSTP) directives on algorithmic fairness, this project evaluates how standard predictive algorithms risk codifying historical medical biases, specifically regarding the diagnosis of cardiovascular disease across varying demographic subgroups (male vs. female cohorts).

We utilize the benchmark **UCI Heart Disease Dataset** to compare a transparent linear baseline (**Logistic Regression**) against an ensemble non-linear model (**XGBoost**). The evaluation is conducted using Microsoft's `Fairlearn` toolkit to quantify disparities such as Demographic Parity Difference and Equalized Odds Difference.

## Key Findings
* **The Accuracy-Fairness Trade-Off:** Logistic Regression achieved a higher nominal accuracy (87.8%) compared to XGBoost (82.2%) but demonstrated marked systemic inequity (Equalized Odds Difference of 0.148 vs. 0.094).
* **Systemic Under-Diagnosis in Women:** The Logistic Regression model yielded a **0.000 False Positive Rate (FPR)** for female patients compared to a 0.148 FPR for male patients. This zero-false-positive phenomenon highlights extreme diagnostic conservatism, systematically missing atypical or borderline cardiac presentations in women.
* **Algorithmic Debiasing:** XGBoost's non-linear architecture successfully captured complex multi-variable interactions, narrowing the demographic gap and reducing the Equalized Odds Difference by 36.5%.

## Repository Structure
* `bias_audit.py`: The primary Python script that fetches the dataset, trains the Logistic Regression and XGBoost models, and generates fairness metrics.
* `fpr_disparity.png`: Visualization of the False Positive Rate disparities between models (generated via matplotlib).
* `Algorithmic_Bias_in_Healthcare_Research_Paper.docx`: The complete, publication-ready academic paper detailing the methodology and clinical implications.  

## Installation & Setup
To run this audit locally, you need Python installed along with the required data science and fairness libraries. 

1. Clone the repository:
   ```bash
   git clone https://github.com/NiloySarker325/Hiring-Algorithms-and-Algorithmic-Bias-in-Healthcare.git
   cd Hiring-Algorithms-and-Algorithmic-Bias-in-Healthcare
   ```

2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn xgboost fairlearn ucimlrepo matplotlib
   ```

## Usage
Run the auditing script directly from your terminal. The script will automatically fetch the UCI dataset via API, train both models, and print the fairness disparities to the console.

```bash
python bias_audit.py
```

## Author
**Niloy Sarker Bappy**  
Department of Computer Science  
Minnesota State University, Mankato

## Acknowledgments
* Data sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).
* Bias auditing performed using [Fairlearn](https://fairlearn.org/).