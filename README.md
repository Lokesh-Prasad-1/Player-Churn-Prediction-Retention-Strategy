Tech Stack:
pandas, numpy – Data manipulation

matplotlib, seaborn – Visualization

scikit-learn – ML modeling & preprocessing

xgboost – Gradient boosting classifier

Key Features:
Synthetic Dataset Generator: Simulates player data including login history, session behavior, in-game purchases, and engagement.

Feature Engineering: Extracts meaningful features like time since last login and scales numerical inputs.

Classification Models: Implements Random Forest and XGBoost classifiers to predict churn.

Model Evaluation: Uses precision, recall, F1-score, and confusion matrix to evaluate predictions.

Feature Importance: Visualizes key factors contributing to player churn.

Retention Strategy Engine: Assigns action plans based on churn risk score thresholds.


Example Output
Random Forest Classification Report:
              precision    recall  f1-score   support
...
Top Churn Indicators:
        feature     importance
0   days_since_last_login    0.38
...
Sample Retention Strategies:
  churn_risk_score                 strategy
0             0.82  Send retention offer + bonus loot box
...


To-Do / Future Improvements:
Integrate with real Steam dataset or API

Deploy as a REST API or interactive dashboard

A/B test strategy effectiveness on real users

Add deep learning models (e.g., LSTM on sequential data)
