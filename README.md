# Titanic Survival Prediction (Kaggle)

An end-to-end machine learning project on the Kaggle Titanic dataset, focused on **EDA-driven feature engineering**, **robust preprocessing pipelines**, and **systematic model benchmarking**.

---

## Objective
Predict passenger survival while demonstrating a complete ML workflow on structured tabular data.

---

## Exploratory Data Analysis (EDA)
EDA was used to **inform feature design**, not just visualisation.

- Survival analysis by:
  - Passenger class, sex, embarkation port
  - Family size and group composition
- Distribution analysis of:
  - Age and fare (stratified by survival)
- Grouped statistics to uncover non-linear patterns
- Correlation analysis on numeric features to detect redundancy

---

## Feature Engineering

### Family & Social Context
- `FamilySize = SibSp + Parch + 1`
- Family grouped into **Alone / Small / Medium / Large**
- Survival trends analysed per family group

### Age & Fare
- Missing ages imputed using **mean age per passenger class**
- 'Fare' missing values filled with dataset mean
- Quantile-based binning (`qcut`) to capture non-linear effects

### Name-Based Features
- Title extraction from passenger names
- Rare titles consolidated into meaningful groups
- Name length computed and binned as a proxy for social status

### Ticket Features
- Ticket prefixes cleaned and normalised
- Ticket frequency counts to capture group travel
- Ticket frequency grouped into semantic buckets

### Cabin Features
- Cabin presence indicator
- Deck letter extraction
- Binary cabin assignment feature

---

## Preprocessing
- Correlation Matrix used to observe the relevance of features
- `ColumnTransformer` used to build reusable pipelines
- One-hot and ordinal encoding for respective categorical features
- Scaling applied where appropriate
- SimpleImputer for fail-safe
- Leakage-safe preprocessing within cross-validation

---

## Models & Evaluation

All models trained using:
- Stratified train/validation split
- 5-fold Stratified Cross-Validation
- `GridSearchCV` for hyperparameter tuning
- Unified preprocessing + model pipelines

### Models Tested
- Logistic Regression
- KNN
- SVM
- Decision Tree
- Random Forest
- Extra Trees
- Naive Bayes
- AdaBoost
- XGBoost

---

## Key Findings
- Tree-based ensemble models performed the strongest
- Feature engineering had a larger impact than model choice
- Social and group-based features were highly predictive
- Pipelines + cross-validation were critical for fair comparison
