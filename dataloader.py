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
                                  '33', '34', '35', '36', '37', '38', '39', 
                                  '40', '41', '42', '43', '44', '45', '46', 
                                  '47', '48', '49', '50', '51', '52', '53', 
                                  '54', '55', '56', '57', '58', '59', '60', 
                                  '61', '62', '63', '64', '65', '66', '67', 
                                  '68', '69', '70', '71', '72', '73', '74', 
                                  '75', '76', '77', '78', '79', '80', '81',
                                  '82', '83', '84', '85', '86', '87', '88', 
                                  '89', '90', '91', '92', '93', '94', '95', 
                                  '96', '97', 'SredniaCena']


    def prepare_sells_data(self, product):
        cols_to_drop = [self.sellsData.columns[1], self.sellsData.columns[2], self.sellsData.columns[-1]]  #usunięcie zbędnych kolumn
        self.sellsData = self.sellsData.drop(cols_to_drop, axis=1)
        cols_to_reverse = [str(i) for i in range(1, 98)]
        cols_present = [col for col in cols_to_reverse if col in self.sellsData.columns]
        cols_reversed = list(reversed(cols_present))
        other_cols = [col for col in self.sellsData.columns if col not in cols_present]
        new_order = other_cols[:1] + cols_reversed
        self.sellsData = self.sellsData[new_order]
        
        filtered_df = self.sellsData[self.sellsData['KodProduktu'] == product] #filtrowanie po wskazanym kodzie
        
        summed = filtered_df[cols_reversed].sum() #sumowanie wyników

        result = pd.DataFrame([summed])
        
        return result

    def load_bom_data_from_file(self, path):
        self.bomData = pd.read_csv(path, sep=';', header=None)



