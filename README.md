# Energy Production Forecasting

A comprehensive time-series forecasting benchmark comparing classical statistical models, machine learning, deep learning, ensemble methods, global vs series-specific forecasting, and zero-shot foundation models.

## Project Overview

This project evaluates multiple forecasting approaches on monthly energy production data across ten energy-production series.

All models are evaluated using the same 24-month out-of-sample holdout period.

The project addresses three main questions:

1. Which forecasting method produces the lowest forecast error?
2. Can a global forecasting model outperform models fitted separately to individual energy series?
3. Can zero-shot foundation models compete with models trained directly on the historical dataset?

## Key Results

- **Best overall model:** DeepAR
- **RMSE:** 0.081
- **Mean Ensemble RMSE:** 0.089
- **Chronos-2 Zero-Shot RMSE:** 0.180
- **TimesFM 3.0 Zero-Shot RMSE:** 0.214
- **Global forecasting wins:** 6 of 10 energy series
- **Series-specific forecasting wins:** 4 of 10 energy series
- **Best conventional machine-learning model:** Recursive Elastic Net / GLMNET

## Forecasting Approaches

### Classical Forecasting

- Seasonal Naive
- Rolling Mean baselines
- ETS
- TBATS
- STL + ETS
- ARIMA
- NNETAR

### Machine Learning

- Elastic Net
- XGBoost
- LightGBM
- Random Forest
- Support Vector Regression

Feature-engineering strategies include calendar features, lag features and hybrid combinations.

### Deep Learning

- DeepAR
- DeepState
- Gaussian Process Forecaster

### Ensemble Methods

- Mean Ensemble
- Median Ensemble
- Weighted Ensemble
- Elastic-Net Stacked Ensemble

### Foundation Models

Two pretrained time-series foundation models are evaluated in a zero-shot setting:

- Amazon Chronos-2
- Google TimesFM 3.0

Neither model is trained specifically on the energy dataset before forecasting.

## Global vs Series-Specific Forecasting

The project compares two forecasting strategies:

**Global forecasting**

A single model learns patterns across all energy-production series.

**Series-specific forecasting**

Separate candidate models are fitted independently to each energy series.

The global approach produced the lowest RMSE for 6 of the 10 series, while local specialization performed better for 4 series.

This suggests a hybrid production strategy in which a strong global model is used by default, with local models retained where they provide meaningful gains.

## Model Benchmark

| Model | RMSE |
|---|---:|
| DeepAR | 0.081 |
| Mean Ensemble | 0.089 |
| Weighted Ensemble | 0.090 |
| Median Ensemble | 0.092 |
| DeepState | 0.106 |
| Recursive GLMNET | 0.140 |
| Chronos-2 Zero-Shot | 0.180 |
| Seasonal Naive | 0.189 |
| TimesFM 3.0 Zero-Shot | 0.214 |

Lower RMSE indicates better forecasting performance.

## Evaluation Metrics

Models are compared using:

- RMSE — Root Mean Squared Error
- MAE — Mean Absolute Error
- MAPE — Mean Absolute Percentage Error

All approaches use the same 24-month holdout period.

## Technologies

- R
- tidyverse
- tidymodels
- timetk
- modeltime
- modeltime.ensemble
- modeltime.resample
- modeltime.gluonts
- xgboost
- LightGBM
- glmnet
- randomForest
- kernlab
- Python via reticulate
- Amazon Chronos-2
- Google TimesFM 3.0
- Quarto

## Project Files

- `Energy_Production_Forecasting_FINAL.R` — complete modelling workflow
- `Energy_Production_Forecasting.qmd` — Quarto portfolio report
- `Energy_Production_Forecasting.html` — rendered interactive report
- `outputs/` — benchmark and evaluation results
- `data/` — source and foundation-model evaluation data

## Reproducibility

The modelling workflow and report-generation workflow are separated.

The full R script trains, evaluates and saves the models and benchmark outputs.

The Quarto report reads the saved outputs rather than retraining every model during rendering. This makes the report faster and more stable while preserving the complete modelling workflow separately.

## Main Findings

1. DeepAR produced the best overall forecast accuracy.
2. Simple ensembles performed almost as well as the best individual model.
3. Global forecasting outperformed local forecasting for most energy series.
4. Recursive Elastic Net was the strongest conventional machine-learning model.
5. Chronos-2 outperformed TimesFM 3.0 in the zero-shot foundation-model benchmark.
6. More complex models did not automatically outperform simpler or ensemble approaches.
7. A hybrid global/local forecasting architecture appears promising.

## Author

**Marcin Szalacha**

Data Analytics | Business Intelligence | Machine Learning | Forecasting
