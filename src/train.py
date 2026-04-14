import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")


# creating the model
model = RandomForestClassifier(class_weight='balanced')

# fit the training data
model.fit(X_train,y_train)

# now predit the data
y_pred = model.predict(X_test)


# saving the model to churn_model.pkl
# by using the library joblib

joblib.dump(model, "models/churn_model.pkl")
print("model saved succesfully")