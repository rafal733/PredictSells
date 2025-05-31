import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self):
        self.sellsData = pd.DataFrame
        self.productsList = []


    def load_sells_data_from_file(self, path):
        self.sellsData = pd.read_csv(path, sep=';', header=None)
        self.sellsData.columns = ['KodProduktu', 'JM', 'Klient', '0', '1', '2', '3',
                                  '4', '5', '6', '7', '8', '9', '10', '11',
                                  '12', '13', '14', '15', '16', '17', '18',
                                  '19', '20', '21', '22', '23', '24', '25',
                                  '26', '27', '28', '29', '30', '31', '32',
                                  '33', '34', '35', '36', '37', '38', '39',
                                  '40', '41', '42', '43', '44', '45', '46',
                                  '47', '48', '49', '50', '51', '52', '53',
                                  '54', '55', '56', '57', '58', '59', '60',
                                  '61', '62', '63', '64', '65', '66', '67',
                                  '68', '69', '70', '71', '72', '73', '74',
                                  '75', '76', '77', '78', '79', '80', '81',
                                  '82', '83', '84', '85', '86', '87', '88',
                                  '89', '90', '91', '92', '93', '94', '95',
                                  '96', '97']

    def prepare_sells_data(self):
        cols_to_drop = [self.sellsData.columns[1], self.sellsData.columns[3]]
        self.sellsData = self.sellsData.drop(cols_to_drop, axis=1)
        cols_to_reverse = [str(i) for i in range(1, 97)]
        cols_present = [
            col for col in cols_to_reverse if col in self.sellsData.columns]
        cols_reversed = list(reversed(cols_present))
        other_cols = [
            col for col in self.sellsData.columns if col not in cols_present]
        new_order = other_cols[:2] + cols_reversed
        self.sellsData = self.sellsData[new_order]
        
        allowed_prefixes = ('0', '1', '2', '3', '4', '5', '7', '9', 'UW', 'US')
        self.sellsData = self.sellsData[self.sellsData['KodProduktu'].astype(str).str.startswith(allowed_prefixes)]

        return self.sellsData

    def prepare_products_list(self):
        uniquelist = self.sellsData['KodProduktu'].unique().tolist()
        self.productsList = uniquelist
        return self.productsList


    def prepare_training_data(self):
        month_columns = [str(i) for i in range(96, 0, -1)]

        df_agg = self.sellsData.groupby("KodProduktu")[month_columns].sum().reset_index()   #Pogrupowanie danych ze względu na kod

        df_long = df_agg.melt(id_vars="KodProduktu", var_name="miesiac_ago", value_name="sprzedaz") #Przekształcenie na długi format
        df_long["miesiac_ago"] = df_long["miesiac_ago"].astype(int)
        df_long["czas"] = 97 - df_long["miesiac_ago"]

        lags = list(range(1, 37)) #tworzenie lagów z 36 ostatnich miesięcy
        for lag in lags:
            df_long[f"lag_{lag}"] = df_long.groupby("KodProduktu")["sprzedaz"].shift(lag)

        # Tworzenie cech dodatkowych
        # Średnie 3 i 12 miesięczne sprzedaży
        df_long["rolling_mean_3"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.rolling(window=3).mean())
        df_long["rolling_mean_12"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.rolling(window=12).mean())

        # Trendy 3 i 12 miesięczne
        df_long["trend_3"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.diff(3))
        df_long["trend_12"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.diff(12))

        # Odchylenia 3 i 12 miesięczne
        df_long["std_3"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.rolling(3).std())
        df_long["std_12"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(lambda x: x.rolling(12).std())

        # Śumy sprzedaży za 3 ostatnie lata
        for i in range(1, 4):
            df_long[f"suma_rok_{i}"] = df_long.groupby("KodProduktu")["sprzedaz"].transform(
                lambda x: x.shift(i * 12).rolling(12).sum()
            )

        # 6. Usunięcie braków
        df_lags = df_long.dropna().copy()

        # 7. Kodowanie produktu
        le = LabelEncoder()
        df_lags["KodProduktu_enc"] = le.fit_transform(df_lags["KodProduktu"])

        # 8. Dane treningowe
        additional_features = ["rolling_mean_3", "rolling_mean_12", 
                        "trend_3", "trend_12", 
                        "std_3", "std_12",
                        "suma_rok_1", "suma_rok_2", "suma_rok_3"
        ]
        X = df_lags[[f"lag_{l}" for l in lags] + additional_features + ["KodProduktu_enc"]]
        y = df_lags["sprzedaz"]
        return X, y, df_lags, le