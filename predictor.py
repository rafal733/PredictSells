import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

class Predictor:
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )

    def model_train(self, X, y):
        self.model.fit(X, y)

    def forecasting(self, product, df_lags, le):
        product_enc = le.transform([product])[0]

        df_product = df_lags[df_lags["KodProduktu"] == product].sort_values("czas", ascending=False)
        last_values = df_product.head(36)["sprzedaz"].tolist()[::-1]

        preds = []

        for i in range(3):
            lag_features = last_values[-36:]
            rolling_mean_3 = pd.Series(last_values[-3:]).mean()
            rolling_mean_12 = pd.Series(last_values[-12:]).mean()
            trend_3 = last_values[-1] - last_values[-4] if len(last_values) >= 4 else 0
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
            y_pred = max(0, y_pred)
            preds.append(y_pred)
            last_values.append(y_pred)

            print(f"Prognozowana sprzedaż produktu {product} za miesiąc {i+1} od teraz: {y_pred:.2f}")
        
        history_length = 36
        future_steps = len(preds)

        sprzedaz_historyczna = last_values[-(history_length + future_steps):-future_steps]
        sprzedaz_prognozowana = preds

        # Oś czasu: -35 do -1 (historia) i 1 do N (prognozy)
        czas_axis_hist = list(range(-history_length + 1, 1))
        czas_axis_pred = list(range(1, future_steps + 1))

        plt.figure(figsize=(12, 6))
        plt.plot(czas_axis_hist, sprzedaz_historyczna, label="Historyczna sprzedaż", marker='o')
        plt.plot(czas_axis_pred, sprzedaz_prognozowana, label="Prognoza", marker='x', linestyle='--', color='red')

        plt.axvline(x=0, color='gray', linestyle='--', label='Teraz')
        plt.title(f"Sprzedaż i prognoza produktu {product}")
        plt.xlabel("Miesiące (0 = teraz, ujemne = przeszłość, dodatnie = przyszłość)")
        plt.ylabel("Wielkość sprzedaży")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
