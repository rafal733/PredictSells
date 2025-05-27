import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Wczytaj dane
df = pd.read_csv('sprzedaz.csv', sep=';', header=None)
df = df[df['']]

# Kolumny miesięczne
monthly_sales = df.iloc[:, 2:-1]  # pomiń kod produktu, klienta i ostatnią kolumnę (łączna suma?)

# Tworzymy zestawy cech z poprzednich 12 miesięcy, cel to sprzedaż w kolejnym miesiącu
X = monthly_sales.iloc[:, :-1]  # miesiące 1-11
y = monthly_sales.iloc[:, -1]   # miesiąc 12 (do przewidzenia)

# Trening/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
