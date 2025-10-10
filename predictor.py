import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


class Predictor:
    def __init__(
        self, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        print(
            "n_estimators=",
            n_estimators,
            "max_depth=",
            max_depth,
            "learning_rate=",
            learning_rate,
            "random_state=",
            random_state,
        )

    def model_train(self, X, y):
        self.model.fit(X, y)

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print("--- Ocena Modelu ---")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R^2: {r2:.4f}")
        print("--------------------")
        return rmse, mae, r2

    def forecasting(self, product, df_lags, le):
        if product not in le.classes_:
            print(f"Produkt {product} nie był użyty podczas treningu. Brak prognozy.")

            df_product = df_lags[df_lags["KodProduktu"] == product].sort_values(
                "czas", ascending=False
            )
            last_values = df_product.head(36)["sprzedaz"].tolist()[::-1]

            if len(last_values) < 1:
                print("Brak danych historycznych do wyświetlenia.")
                return

            if len(last_values) < 36:
                last_values = [0] * (36 - len(last_values)) + last_values

            czas_axis_hist = list(range(-35, 1))

            plt.figure(figsize=(12, 6))
            plt.plot(
                czas_axis_hist, last_values, label="Historyczna sprzedaż", marker="o"
            )
            plt.axvline(x=0, color="gray", linestyle="--", label="Teraz")
            plt.title(f"Tylko dane historyczne – produkt {product}")
            plt.xlabel("Miesiące (0 = teraz, ujemne = przeszłość)")
            plt.ylabel("Wielkość sprzedaży")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            return

        product_enc = le.transform([product])[0]
        df_product = df_lags[df_lags["KodProduktu"] == product].sort_values(
            "czas", ascending=False
        )
        last_values = df_product.head(36)["sprzedaz"].tolist()[::-1]
        preds = []
        ostatni_miesiac = df_product.iloc[0]["numer_miesiaca"]

        for i in range(3):
            lag_features = last_values[-36:]
            rolling_mean_3 = pd.Series(last_values[-3:]).mean()
            rolling_mean_6 = pd.Series(last_values[-6:]).mean()
            trend_24 = (
                pd.Series(last_values).diff().rolling(24).mean().iloc[-1]
                if len(last_values) >= 25
                else 0
            )
            trend_6 = (
                pd.Series(last_values).diff().rolling(6).mean().iloc[-1]
                if len(last_values) >= 7
                else 0
            )
            std_3 = pd.Series(last_values[-3:]).std()
            std_6 = pd.Series(last_values[-6:]).std()
            sum_year_1 = sum(last_values[-12:])
            sum_year_2 = sum(last_values[-24:-12])
            sum_year_3 = sum(last_values[-36:-24])
            numer_miesiaca = (ostatni_miesiac + 1) % 12
            numer_miesiaca = 12 if numer_miesiaca == 0 else numer_miesiaca

            x_input = lag_features + [
                rolling_mean_3,
                rolling_mean_6,
                trend_24,
                trend_6,
                std_3,
                std_6,
                sum_year_1,
                sum_year_2,
                sum_year_3,
                product_enc,
                numer_miesiaca,
            ]

            y_pred = self.model.predict([x_input])[0]
            y_pred = max(0, y_pred)
            preds.append(y_pred)
            last_values.append(y_pred)

            print(
                f"Prognozowana sprzedaż produktu {product} za miesiąc {i+1} od teraz: {y_pred:.2f}"
            )

        history_length = 36
        future_steps = len(preds)

        historic_sell = last_values[-(history_length + future_steps) : -future_steps]
        forecast_sell = preds

        time_axis_hist = list(range(-history_length + 1, 1))
        time_axis_pred = list(range(1, future_steps + 1))

        plt.figure(figsize=(12, 6))
        plt.plot(
            time_axis_hist, historic_sell, label="Historyczna sprzedaż", marker="o"
        )
        plt.plot(
            time_axis_pred,
            forecast_sell,
            label="Prognoza",
            marker="x",
            linestyle="--",
            color="red",
        )
        plt.axvline(x=0, color="gray", linestyle="--", label="Teraz")
        plt.title(f"Sprzedaż i prognoza produktu {product}")
        plt.xlabel("Miesiące (0 = teraz, ujemne = przeszłość, dodatnie = przyszłość)")
        plt.ylabel("Wielkość sprzedaży")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def forecasting_www(self, product, df_lags, le):
        if product not in le.classes_:
            text = f"Brak danych lub prognozy dla produktu {product}."
            return None, None, text

        product_enc = le.transform([product])[0]
        df_product = df_lags[df_lags["KodProduktu"] == product].sort_values(
            "czas", ascending=False
        )
        last_values = df_product.head(36)["sprzedaz"].tolist()[::-1]
        preds = []

        ostatni_miesiac = df_product.iloc[0]["numer_miesiaca"]

        if sum(last_values[-6:]) == 0:
            text = f"Brak sprzedaży przez ostatnie 6 miesięcy dla produktu {product}. Prognoza = 0."
            # Przygotuj dane do wykresu: historia + 3 prognozy zero
            history_length = 36
            future_steps = 3
            historic_sell = last_values[-history_length:]
            forecast_sell = [0] * future_steps
            return historic_sell, forecast_sell, text

        for i in range(3):
            lag_features = last_values[-36:]
            rolling_mean_3 = pd.Series(last_values[-3:]).mean()
            rolling_mean_6 = pd.Series(last_values[-6:]).mean()
            trend_24 = (
                pd.Series(last_values).diff().rolling(24).mean().iloc[-1]
                if len(last_values) >= 25
                else 0
            )
            trend_6 = (
                pd.Series(last_values).diff().rolling(6).mean().iloc[-1]
                if len(last_values) >= 7
                else 0
            )
            std_3 = pd.Series(last_values[-3:]).std()
            std_6 = pd.Series(last_values[-6:]).std()
            sum_year_1 = sum(last_values[-12:])
            sum_year_2 = sum(last_values[-24:-12])
            sum_year_3 = sum(last_values[-36:-24])
            numer_miesiaca = (ostatni_miesiac + i) % 12
            numer_miesiaca = 12 if numer_miesiaca == 0 else numer_miesiaca

            x_input = lag_features + [
                rolling_mean_3,
                rolling_mean_6,
                trend_24,
                trend_6,
                std_3,
                std_6,
                sum_year_1,
                sum_year_2,
                sum_year_3,
                product_enc,
                numer_miesiaca,
            ]

            y_pred = self.model.predict([x_input])[0]
            y_pred = max(0, y_pred)
            preds.append(y_pred)
            last_values.append(y_pred)

        text = f"Prognoza sprzedaży produktu {product} za kolejne 3 miesiące."
        history_length = 36
        future_steps = len(preds)
        historic_sell = last_values[-(history_length + future_steps) : -future_steps]
        forecast_sell = preds

        return historic_sell, forecast_sell, text

    def forecast_all_products_to_csv(
        self, df_lags, products_list, le, output_path="prognozy.csv"
    ):
        forecast_results = []

        for product_code in products_list:
            try:
                if product_code not in le.classes_:
                    preds = [0, 0, 0]
                else:
                    product_enc = le.transform([product_code])[0]
                    df_product = df_lags[
                        df_lags["KodProduktu"] == product_code
                    ].sort_values("czas", ascending=False)
                    last_values = df_product.head(36)["sprzedaz"].tolist()[::-1]
                    ostatni_miesiac = df_product.iloc[0]["numer_miesiaca"]

                    if sum(last_values[-6:]) == 0:
                        preds = [0, 0, 0]
                    else:
                        preds = []
                        for i in range(3):
                            lag_features = last_values[-36:]
                            rolling_mean_3 = np.mean(last_values[-3:])
                            rolling_mean_6 = np.mean(last_values[-6:])
                            trend_24 = (
                                pd.Series(last_values)
                                .diff()
                                .rolling(24)
                                .mean()
                                .iloc[-1]
                                if len(last_values) >= 25
                                else 0
                            )
                            trend_6 = (
                                pd.Series(last_values).diff().rolling(6).mean().iloc[-1]
                                if len(last_values) >= 7
                                else 0
                            )
                            std_3 = np.std(last_values[-3:])
                            std_6 = np.std(last_values[-6:])
                            sum_year_1 = sum(last_values[-12:])
                            sum_year_2 = sum(last_values[-24:-12])
                            sum_year_3 = sum(last_values[-36:-24])
                            numer_miesiaca = (ostatni_miesiac + i) % 12
                            numer_miesiaca = (
                                12 if numer_miesiaca == 0 else numer_miesiaca
                            )

                            x_input = lag_features + [
                                rolling_mean_3,
                                rolling_mean_6,
                                trend_24,
                                trend_6,
                                std_3,
                                std_6,
                                sum_year_1,
                                sum_year_2,
                                sum_year_3,
                                product_enc,
                                numer_miesiaca,
                            ]

                            y_pred = self.model.predict([x_input])[0]
                            y_pred = max(0, y_pred)
                            preds.append(round(y_pred, 2))
                            last_values.append(y_pred)

                forecast_results.append(
                    {
                        "KodProduktu": "'" + product_code,
                        "Prognoza_M1": str(preds[0]).replace(".", ","),
                        "Prognoza_M2": str(preds[1]).replace(".", ","),
                        "Prognoza_M3": str(preds[2]).replace(".", ","),
                    }
                )

            except Exception as e:
                print(f"⚠️ Błąd dla produktu {product_code}: {e}")
                continue

        df_forecast = pd.DataFrame(forecast_results)
        df_forecast.to_csv(output_path, index=False, sep=";")
        print(
            f"\n✅ Plik '{output_path}' został zapisany z prognozami dla {len(df_forecast)} produktów."
        )
