import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

class Predictor:
    
    def __init__(self, le, n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
        self.le = le
    
    def model_train(self, X, y):
        self.model.fit(X, y)    
    
    def forecasting(self, product, df_lags):
        product_enc = self.le.transform([product])[0]

        df_product = df_lags[df_lags["KodProduktu"] == product].sort_values("czas", ascending=False)
        last_values = df_product.head(36)["sprzedaz"].tolist()[::-1]  # najnowsze jako ostatnie

        preds = []
        for i in range(3):
            lag_features = last_values[-36:]
            rolling_mean_3 = pd.Series(last_values[-3:]).mean()
            rolling_mean_12 = pd.Series(last_values[-12:]).mean()
            trend_3 = last_values[-1] - last_values[-4]
            trend_12 = last_values[-1] - last_values[-13] if len(last_values) >= 13 else 0
            std_3 = pd.Series(last_values[-3:]).std()
            std_12 = pd.Series(last_values[-12:]).std()

            suma_rok_1 = sum(last_values[-12:])
            suma_rok_2 = sum(last_values[-24:-12])
            suma_rok_3 = sum(last_values[-36:-24])

            x_input = lag_features + [
                                    rolling_mean_3, rolling_mean_12,
                                    trend_3, trend_12,
                                    std_3, std_12,
                                    suma_rok_1, suma_rok_2, suma_rok_3,
                                    product_enc
                                ]

            y_pred = self.model.predict([x_input])[0]
            preds.append(y_pred)
            last_values.append(y_pred)

        # 11. Wynik
        for i, p in enumerate(preds):
            print(f"Prognozowana sprzedaż produktu {product} za miesiąc {i+1} od teraz: {p:.2f}")