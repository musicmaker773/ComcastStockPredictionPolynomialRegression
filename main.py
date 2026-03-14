from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("comcaststock.csv")
df['Price'] = df['Price'].astype("float")

input_data = []
output_data = []

for i in range(len(df)):
    input_data.append([i])

for i in range(len(df)):
    output_data.append([df['Price'][i]])

model = make_pipeline(PolynomialFeatures(3), Ridge())
model.fit(input_data, output_data)

predict_date = []
predict_data = []

for i in range(7):
    predict_date.append([len(df) + i])
    predict_data.append(model.predict([[len(df) + i]])[0])

model_data = []
for i in range(len(df)):
    model_data.append(model.predict([[i]])[0])

orig_dates = df['Date'].tolist()
orig_price = df['Price'].tolist()

predict_dates = [
    "October 21",
    "October 22",
    "October 23",
    "October 24",
    "October 27",
    "October 28",
    "October 29"
]

# Drawing Graph
plt.title("Market Prediction of Comcast(CMSCA)")
plt.figure(figsize = (20,5))

plt.xticks(rotation = 90)

plt.xlabel("Date")
plt.ylabel("Price")

plt.plot(orig_dates, orig_price)
plt.plot(orig_dates, model_data)
plt.plot(predict_dates, predict_data)

plt.savefig("ComcastPrediction.png")

plt.legend()
plt.show()

for i in range(len(predict_data)):
    print(predict_dates[i] + ": " + str(predict_data[i]))
