import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('GS.csv')

df['Date'] = pd.to_datetime(df['Date'], format= '%d/%m/%Y %H:%M:%S')

df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

df = df.drop(columns=['Date'])

df['Change'] = df['Close'].diff().shift(-1)
df['Change'] = df['Change'].apply(lambda x: 1 if x > 0 else(-1 if x < 0 else 0))
df.dropna()

X = df[['Day', 'Month', 'Year']]
Y = df['Change']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=69)
rf_model.fit(X_train, Y_train)

Y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_report(Y_test, Y_pred, zero_division=1))

joblib.dump(rf_model,"random_forest_model.pkl")
