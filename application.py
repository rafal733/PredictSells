from dataloader import DataLoader
from predictor import Predictor

# 1. Wczytaj dane
loader = DataLoader()
loader.load_sells_data_from_file('sprzedaz.csv')
loader.prepare_sells_data()
loader.prepare_products_list()

# 2. Przygotuj dane treningowe
df_lags, X, y = loader.prepare_training_data()

# 3. Trening modelu
predictor = Predictor()
predictor.model_train(X, y)

while True:
    product_code = input("Podaj kod produktu do prognozy (lub wpisz 'exit' aby zakończyć): ").strip()
    
    if product_code.lower() == 'exit':
        print("Zakończono prognozowanie.")
        break

    if product_code not in loader.productsList:
        print("❌ Nieprawidłowy kod produktu lub brak danych. Spróbuj ponownie.")
        continue

    try:
        predictor.forecasting(product_code, df_lags, loader.le)
    except Exception as e:
        print(f"Błąd podczas prognozy: {e}")