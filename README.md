# Student Mental Health Prediction — Composite Scoring Framework

Code repository accompanying the manuscript **"Composite Scoring Framework for Multi-Algorithm Student Depression Prediction"**, submitted to *Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)*.

This repository contains the machine learning pipeline used to predict depression risk among university students using four base classifiers (Logistic Regression, SVM, Random Forest, XGBoost) and two ensemble schemes (Majority Voting and Stacking), evaluated under 5-Fold Stratified Cross-Validation with Paired T-Test significance testing and a nine-dimension Composite Score combining classification performance and computational efficiency (including Cyclomatic Complexity).

## Dataset

This study uses the **"Students' Mental Health in a Multicultural Environment"** dataset, a freely available public dataset collected via a web-based survey (October–December 2018). It includes:
- **Patient Health Questionnaire (PHQ-9)** — depressive symptoms (target label: `Dep`)
- **Social Connectedness Scale (SCS)**
- **Acculturative Stress Scale for International Students (ASSISS)** — 7 sub-dimensions
- **General Health Help-Seeking Questionnaire (GHSQ)**
- Sociodemographic variables

After preprocessing, the working dataset contains **268 respondents**, **27 predictor features**, and 1 binary target (`Dep`: 172 non-depressed / 96 depressed).

The dataset itself is **not included** in this repository. Place `data.csv` in your own storage (e.g., Google Drive) and update the `data_path` variable in each notebook accordingly (see [Reproducibility Notes](#reproducibility-notes) below).

## Repository Structure

| Notebook | Purpose |
|---|---|
| `1_MentalHealthPrediction_EDA_.ipynb` | Exploratory data analysis: missing values, class distribution, feature overview |
| `2_MentalHealthPrediction(clean_dataset).ipynb` | Data cleaning, feature selection, label encoding, Z-score standardization |
| `3_Mental_Health_Experiment(Fix_Thesis).ipynb` | Full experimental pipeline: model training (LR, SVM, RF, XGBoost), ensemble construction (Majority Voting, Stacking), 5-Fold Stratified Cross-Validation, Paired T-Test significance testing, computational efficiency measurement, and Composite Score calculation — corresponds to **Tables 4–6** and **Figures 2–9** in the manuscript |

## Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
xgboost
scipy
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn
```

## How to Run

1. Open the notebooks in Google Colab (recommended) or a local Jupyter environment, in order: `1` → `2` → `3`.
2. Upload `data.csv` to your own Google Drive and mount it in Colab, or place it in a local `data/` folder.
3. Update the `data_path` variable at the top of each notebook to point to your dataset location.
4. Run all cells sequentially. Notebook 3 reproduces the classification performance, computational efficiency, and Composite Score results reported in the manuscript.

## Reproducibility Notes

- Notebooks reference the dataset via a local Colab path (e.g., `/content/drive/MyDrive/03 Dataset/data.csv`). This path is specific to the original authors' Google Drive and will not resolve on another machine — update it to your own dataset location before running.
- A fixed `random_state = 42` is used throughout for reproducibility of data splits, model initialization, and cross-validation folds.
- All preprocessing (scaling) and efficiency measurements (training time, inference time, peak RAM, CPU utilization) are performed strictly within each cross-validation fold to avoid data leakage.

## Citation

If you use this code, please cite the corresponding manuscript (citation details to be updated upon publication in Jurnal RESTI).

## License

This code is released for academic and research purposes. Please contact the authors regarding reuse beyond academic citation.
