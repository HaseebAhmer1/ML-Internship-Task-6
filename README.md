# 🩺 Diabetes Classification Algorithms Comparison

## 🎯 Objective
Build and compare multiple classification algorithms that predict diabetes onset from patient diagnostic measurements, while explicitly handling class imbalance, tuning the decision threshold around a cost-sensitive business scenario, and identifying the best-performing model for deployment.

## 📊 Dataset Information
- **Source file:** `diabetes.csv` (Pima Indians Diabetes dataset)
- **Target variable:** `Outcome` — binary indicator of diabetes diagnosis (34.9% positive)
- **Features:** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- **Preprocessing checks performed:** duplicate detection, and inspection for the classic "zero as missing value" issue in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` (with within-class median imputation defined for any zero-placeholders found)
- **Business framing:** cost-sensitive diabetes screening — a false positive costs ~$20 (unnecessary follow-up test), a false negative costs ~$200 (downstream cost of an undiagnosed patient)

## ✅ Tasks & Outcomes

**1. Data Processing**
- Loaded the dataset and ran the zero-as-missing-value check and duplicate removal
- Performed EDA: class balance plot, per-feature distribution histograms segmented by outcome, and a correlation heatmap
- Split data into training (80%) and testing (20%) sets using a stratified split
- Scaled numerical features with `StandardScaler`, fit only on the training set to avoid data leakage

**2. Class Imbalance Handling**
- Implemented **SMOTE from scratch** (nearest-neighbour interpolation) to avoid an external dependency
- Compared it against `class_weight='balanced'` and training on the raw imbalanced data

**3. Baseline Models**
- Evaluated a majority-class baseline and a stratified-random baseline as trivial lower bounds

**4. Multi-Classifier Comparison Framework**
- Trained and evaluated 5 classifiers — **Logistic Regression, Decision Tree, Random Forest, SVM (RBF kernel), KNN** — across 3 imbalance-handling strategies (14 model/strategy combinations total)
- Compared all combinations on accuracy, precision, recall, F1, ROC-AUC, and PR-AUC

**5. Evaluation & Model Selection**
- Identified the best imbalance strategy per model and visualized key metrics in a grouped bar chart
- Plotted ROC and Precision-Recall curves for the best strategy per model
- Selected **Random Forest** for deployment based on ranking quality (ROC-AUC / PR-AUC), not just F1 at the default threshold

**6. Cost-Sensitive Threshold Tuning**
- Grid-searched the decision threshold that minimizes total business cost (`$20` × false positives + `$200` × false negatives)
- Compared confusion matrices and total cost at the default (0.5) vs. optimal threshold
- Quantified the cost savings from threshold tuning

## 🛠️ Key Skills Demonstrated
- Exploratory data analysis and data visualization (`matplotlib`, `seaborn`)
- Leakage-safe preprocessing pipelines with `scikit-learn` (`StandardScaler`, stratified train/test split)
- Custom implementation of SMOTE from first principles using `NearestNeighbors`
- Classification modeling: Logistic Regression, Decision Tree, Random Forest, SVM, KNN
- Model evaluation using precision, recall, F1, ROC-AUC, and PR-AUC
- ROC and Precision-Recall curve analysis for ranking-quality comparison across models
- Cost-sensitive threshold optimization tied to a concrete business scenario
- Model comparison and selection based on multiple, sometimes conflicting, evaluation criteria

## 💡 Key Learnings
- The best model by F1 at the default 0.5 threshold (Decision Tree) was not the best model overall — Random Forest's superior ROC-AUC/PR-AUC made it the better choice once the threshold was going to be tuned anyway.
- Class imbalance handling (SMOTE, `class_weight='balanced'`) meaningfully improved almost every model compared to training on raw imbalanced data.
- A model's raw accuracy or F1 score can be misleading in the presence of asymmetric business costs — the "right" threshold depends on the relative cost of false positives vs. false negatives, not on 0.5 by default.
- Models with coarse, discrete probability outputs (like a shallow Decision Tree) are poorly suited to threshold tuning, even if they score well at one fixed threshold — smooth, well-calibrated probabilities (like Random Forest's) matter for this kind of optimization.
- Establishing trivial baselines (majority-class, stratified-random) is essential to confirm that a model's performance reflects real learned signal, not just the natural class balance of the data.
