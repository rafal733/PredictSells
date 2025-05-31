from dataloader import DataLoader
from predictor import Predictor
import pandas as pd
import numpy as np
import xgboost as xgb


loader = DataLoader()

loader.load_sells_data_from_file('sprzedaz.csv')
df = loader.prepare_sells_data()
productslist = loader.prepare_products_list()

X, y, df_lags, le = loader.prepare_training_data()


predictor = Predictor(le=le)

predictor.model_train(X, y)

predictor.forecasting('0001', df_lags)
predictor.forecasting('0015', df_lags)
predictor.forecasting('0017', df_lags)