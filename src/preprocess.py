import pandas as pd

df = pd.read_csv("data/raw/Churn_Modelling.csv")

# dropping the useless columns

df = df.drop(columns=["RowNumber","CustomerId", "Surname"])


# Encoding categorical data into numeric values 
