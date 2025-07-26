import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


#Generate Synthetic Data
def generate_churn_dataset(n=5000):
    data = []
    today = datetime.today()

    for i in range(n):
        user_id = f"user_{i}"
        last_login = today - timedelta(days=np.random.randint(0, 60))
        total_play_time = np.random.randint(1, 500)
        avg_session_length = np.random.uniform(10, 90)
        num_sessions = np.random.randint(1, 100)
        num_friends = np.random.randint(0, 50)
        num_purchases = np.random.randint(0, 20)
        level_reached = np.random.randint(1, 100)
        churn_label = 1 if (today - last_login).days > 30 else 0

        data.append([
            user_id, last_login.date(), total_play_time, avg_session_length,
            num_sessions, num_friends, num_purchases, level_reached, churn_label
        ])

    return pd.DataFrame(data, columns=[
        "user_id", "last_login_date", "total_play_time", "avg_session_length",
        "num_sessions", "num_friends", "num_purchases", "level_reached", "churn_label"
    ])


df = generate_churn_dataset()

#Feature Engineering
df['last_login_date'] = pd.to_datetime(df['last_login_date'])
df['days_since_last_login'] = (datetime.today() - df['last_login_date']).dt.days
df.drop(['user_id', 'last_login_date'], axis=1, inplace=True)

# Visualize class distribution
sns.countplot(data=df, x='churn_label')
plt.title("Churn Distribution")
plt.show()

#Prepare Data for Modeling
X = df.drop("churn_label", axis=1)
y = df["churn_label"]

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#Model Training & Evaluation

#Model 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_preds))

#Model 2: XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))

# Feature Importance
feature_names = df.drop("churn_label", axis=1).columns
importances = rf_model.feature_importances_

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print("\nTop Churn Indicators:")
print(feat_imp)

# Plot
sns.barplot(data=feat_imp, x='importance', y='feature')
plt.title("Feature Importance (Random Forest)")
plt.show()

# Retention Strategy Recommendations
df["churn_risk_score"] = rf_model.predict_proba(scaler.transform(df.drop("churn_label", axis=1)))[:, 1]


# Example strategy output
def recommend_strategy(row):
    if row['churn_risk_score'] > 0.7:
        return "Send retention offer + bonus loot box"
    elif row['churn_risk_score'] > 0.5:
        return "Send re-engagement email"
    else:
        return "No action needed"


df["strategy"] = df.apply(recommend_strategy, axis=1)
print("\nSample Retention Strategies:")
print(df[["churn_risk_score", "strategy"]].head(10))
