import numpy as np
import pandas as pd
from os import getcwd, listdir
import matplotlib.pyplot as plt
import seaborn as sns
from jupyterthemes import jtplot
sns.set()
jtplot.style(theme="monokai")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Data Path
data_path = getcwd() + "/fahrenheit_celsius"
listdir(data_path)

df = pd.read_csv(f"{data_path}/{listdir(data_path)[-1]}")

X = df['celsius']
y = df['fahrenheit']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

X_train = np.array(X_train).reshape(-1,1)
Y_train = np.array(Y_train)

model = LinearRegression()

model.fit(X_train, Y_train)

X_test = np.array(X_test).reshape(-1,1)

Y_pred = model.predict(X_test)

r2 = r2_score(Y_test, Y_pred)
r2

model.coef_
model.intercept_

def model(x):
    return 1.8*x + round(31.99999999, 2)

plt.figure(figsize=(10,10))
plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, model(X_test), 'b-')
plt.title("Line of Best Fit")
plt.savefig(f"{data_path}/reg_model.png")
