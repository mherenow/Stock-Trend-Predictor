import joblib
import pandas as pd

model = joblib.load('random_forest_model.pkl')

date = input("Enter the date for prediction(DD/MM/YYYY): ")
date_list = date.split('/')
df = pd.DataFrame([date_list], columns=['Day', 'Month', 'Year'])
df = df.apply(pd.to_numeric)

prediction = model.predict(df)

if prediction == 1:
    print("Price Increases")
elif prediction == -1:
    print("Price Decrease")
else:
    print("Pricr Remains Same")