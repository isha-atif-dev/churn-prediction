import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/raw/Churn_Modelling.csv")

# step 1 :dropping the useless columns

df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])


# step 2 : Encoding categorical data into numeric values
df = pd.get_dummies(data=df, columns=["Geography", "Gender"])
# print(df.columns)


# step 3: separating X and y
# X : all features values except "exited"
# y : only exited which is traget value

X = df.drop(columns="Exited")
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_train.to_csv("data/processed/X_train.csv",index=False)
X_test.to_csv("data/processed/X_test.csv",index=False)
y_train.to_csv("data/processed/y_train.csv",index=False)
y_test.to_csv("data/processed/y_test.csv",index=False)
