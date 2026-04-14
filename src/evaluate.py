# Evaluating the model to check if its perform well or not 

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

model = joblib.load("models/churn_model.pkl")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

y_pred = model.predict(X_test)
# Checking accuracy
accuracy = accuracy_score(y_test,y_pred)

print(f'Accuray {accuracy}')
# got 80 percent accuracy but accuracu alone could be misleading because of the class imbalance 

# so we will also check Recall and Precision

# Precision : out of all customers who were predicted as they will leave _ how many actually left
# Recall : out of all customers who left , did model catch all who left 

recall = recall_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
f1_Score = f1_score(y_test,y_pred)

print(f'Recall {recall}')
print(f'precision {precision}')
print(f'F1_Score {f1_Score}')


# Accuracy 86.75% — looks great on surface 

# Precision 77% — when our model says "this customer will leave", it's right 77% of the time. That's decent.

# Recall 46% — this is the problem 

# Out of all customers who actually left — our model only caught 46% of them. That means it missed 54% of real churners.

# F1 Score 0.57 — confirms the model is average overall despite high accuracy.


'''
The fix for this is called class balancing
'''