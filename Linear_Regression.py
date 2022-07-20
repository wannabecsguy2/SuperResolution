import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

df = pd.read_excel('LR_Data.xlsx')
X, y_mango, y_orange = df[['Temperature', 'Rainfall', 'Humidity']], df[['Mangoes']], df[['Oranges']]
X, y_mango, y_orange = np.array(X), np.array(y_mango), np.array(y_orange)

def LR_Pipeline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipe.fit(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    return test_score, pipe
