## Modeling (Risk Classification)

### Objective

The objective of this phase was to build classification models to predict the diabetes stage using lifestyle and demographic features, and to identify the best-performing model for deployment.

---

### Approach

#### Data Preparation

* The target variable was `diabetes_stage`.
* Features were separated from the target variable.
* Categorical variables were converted into numeric format using one-hot encoding.

#### Handling Data Leakage

Initial model results showed extremely high accuracy (approximately 99%), indicating data leakage. To address this issue, the following highly correlated clinical variables were removed:

* diabetes_risk_score
* glucose_fasting
* glucose_postprandial
* hba1c
* insulin_level

This ensured that the model learned from lifestyle and demographic factors rather than direct diagnostic indicators.

---

### Train-Test Split

The dataset was split into:

* 80% training data
* 20% testing data

A fixed random state was used to ensure reproducibility.

---

### Models Developed

#### Decision Tree

The Decision Tree model was used as a baseline model.

* Accuracy: 0.867
* F1 Score: 0.869

#### Random Forest

The Random Forest model improved performance by combining multiple decision trees.

* Accuracy: 0.917
* F1 Score: 0.881

#### XGBoost

The XGBoost model was implemented as a boosting algorithm to achieve higher predictive performance.

* Accuracy: 0.914
* F1 Score: 0.882

---

### Model Selection

Although Random Forest achieved slightly higher accuracy, XGBoost was selected as the final model because it achieved the highest F1 score and provided a better balance across evaluation metrics.

---

### Challenges

* Data leakage led to unrealistically high model performance and required careful feature removal.
* Errors occurred due to incorrect execution order in the Jupyter Notebook, which were resolved by restarting the kernel and running all cells sequentially.
* File path issues prevented the dataset from loading initially and were fixed by correcting directory paths.
* XGBoost installation issues arose due to environment conflicts and were resolved through reinstallation and kernel restart.
* Class imbalance caused some models to perform poorly on certain classes, requiring the use of weighted evaluation metrics.

---

### Conclusion

The modeling phase successfully developed and evaluated multiple classification models. After addressing data leakage and technical challenges, XGBoost was selected as the best-performing model, providing strong and reliable predictive performance.

---
