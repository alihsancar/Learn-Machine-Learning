import pandas as pd
import numpy as np
import pickle

with open("55-diamond_model_complete.pkl" , "rb") as f:
    saved_data = pickle.load(f)

model = saved_data["model"]
X_test_scaled = pd.read_csv("55-testdatascaled.csv")
print(model.predict(X_test_scaled))