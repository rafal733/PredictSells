import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self):
        self.sellsData = pd.DataFrame
        self.bomData = pd.DataFrame

    def load_sells_data_from_file(self, path):
        self.sellsData = pd.read_csv(path, sep=';', header=None)
        self.sellsData.columns = ['KodProduktu', 'Klient', '0', '1', '2', '3', 
                                  '4', '5', '6', '7', '8', '9', '10', '11', 
                                  '12', '13', '14', '15', '16', '17', '18', 
                                  '19', '20', '21', '22', '23', '24', '25', 
                                  '26', '27', '28', '29', '30', '31', '32', 
                                  '33', '34', '35', '36', 'SredniaCena']


    def prepare_sells_data(self, product):
        cols_to_drop = [self.sellsData.columns[2], self.sellsData.columns[-1]]  #usunięcie zbędnych kolumn
        self.sellsData = self.sellsData.drop(cols_to_drop, axis=1)
        cols_to_reverse = [str(1) for i in range(1, 37)]
        cols_present = [col for col in cols_to_reverse if col in self.sellsData.columns]
        cols_reversed = list(reversed(cols_present))
        other_cols = [col for col in self.sellsData.columns if col not in cols_present]
        new_order = other_cols[:2] + cols_reversed
        self.sellsData = self.sellsData[new_order]
        
        filtered_df = self.sellsData[self.sellsData['KodProduktu'] == product]
        
        return filtered_df

    def load_bom_data_from_file(self, path):
        self.bomData = pd.read_csv(path, sep=';', header=None)



