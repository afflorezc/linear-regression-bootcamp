import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('ipc_2025.csv', sep=';')
df.set_index('Mes', inplace=True)

years = df.columns
ipc = [df[year].dropna().to_list() for year in years]

years_num = years.astype(int)
years_vect = [(np.ones(len(df[year].dropna()))*int(year)).tolist() for year in years]
print(years_vect)

print(ipc[0])

x = []
y = []

for i in range(len(years_vect)):
    x = x + years_vect[i]
    y = y + ipc[i]

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

scatter_data = pd.DataFrame({'x':x.flatten(), 'y':y.flatten()})
scatter_data.to_csv('scatter_data.csv')

years_num = np.array(years_num).reshape(-1, 1)
lr = LinearRegression()
lr.fit(x, y)

model_pickle = open('linear_regression.pickle', 'wb')
pickle.dump(lr, model_pickle)
model_pickle.close()


y_pred = lr.predict(years_num)

plt.scatter(x, y, color="#ff2186", sizes=[2.5])
plt.plot(years_num, y_pred)
plt.show()

