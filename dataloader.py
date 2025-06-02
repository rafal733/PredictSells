import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self):
        self.sellsData = pd.DataFrame()
        self.productsList = []
        self.le = LabelEncoder()

    def load_sells_data_from_file(self, path):
        self.sellsData = pd.read_csv(path, sep=';', header=None)
        self.sellsData.columns = ['KodProduktu', 'JM', 'Klient'] + [str(i) for i in range(0, 98)]

    def prepare_sells_data(self):
        cols_to_drop = [self.sellsData.columns[1], self.sellsData.columns[3]]
        self.sellsData = self.sellsData.drop(cols_to_drop, axis=1)

        month_cols = [str(i) for i in range(1, 97)]
        present_months = [col for col in month_cols if col in self.sellsData.columns]
        reversed_months = list(reversed(present_months))
        other_cols = [col for col in self.sellsData.columns if col not in present_months]
        self.sellsData = self.sellsData[other_cols[:2] + reversed_months]

        allowed_prefixes = ('0', '1', '2', '3', '4', '5', '7', '9', 'UW', 'US')
        self.sellsData = self.sellsData[
            self.sellsData['KodProduktu'].astype(str).str.startswith(allowed_prefixes)
        ]
        return self.sellsData

    def prepare_products_list(self):
        self.productsList = self.sellsData['KodProduktu'].unique().tolist()
        return self.productsList

    def prepare_training_data(self):
        month_columns = [str(i) for i in range(96, 0, -1)]

        df_agg = self.sellsData.groupby("KodProduktu")[month_columns].sum().reset_index()
        df_long = df_agg.melt(id_vars="KodProduktu", var_name="miesiac_ago", value_name="sprzedaz")
        df_long["miesiac_ago"] = df_long["miesiac_ago"].astype(int)
        df_long["czas"] = 97 - df_long["miesiac_ago"]

        lags = list(range(1, 37))
        for lag in lags:
            df_long[f"lag_{lag}"] = df_long.groupby("KodProduktu")["sprzedaz"].shift(lag)

        # Cechy dodatkowe
        df_long["rolling_mean_3"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.rolling(3).mean())
        df_long["rolling_mean_6"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.rolling(6).mean())
        df_long["trend_24"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.diff().rolling(24).mean())
        df_long["trend_6"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.diff().rolling(6).mean())
        df_long["std_3"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.rolling(3).std())        
        df_long["std_6"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.rolling(6).std())

        for i in range(1, 4):
            df_long[f"suma_rok_{i}"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(
                lambda x: x.shift(i * 12).rolling(12).sum()
            )

        df_lags = df_long.dropna().copy()

        # 🔒 Filtrowanie: usuń produkty z zerową sprzedażą w ostatnich 6 miesiącach
        def has_recent_sales(group):
            return group.sort_values("czas", ascending=False).head(6)["sprzedaz"].sum() > 0

        df_lags = df_lags.groupby("KodProduktu").filter(has_recent_sales)

        # Kodowanie produktu
        df_lags["KodProduktu_enc"] = self.le.fit_transform(df_lags["KodProduktu"])

        additional_features = [
            "rolling_mean_3", "rolling_mean_6",
            "trend_24", "trend_6",
            "std_3", "std_6",
            "suma_rok_1", "suma_rok_2", "suma_rok_3"
        ]
        X = df_lags[[f"lag_{l}" for l in lags] + additional_features + ["KodProduktu_enc"]]
        y = df_lags["sprzedaz"]

        return df_lags, X, y
