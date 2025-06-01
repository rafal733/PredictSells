from flask import Flask, render_template, request
from dataloader import DataLoader
from predictor import Predictor

app = Flask(__name__)

loader = DataLoader()
loader.load_sells_data_from_file("sprzedaz.csv")
loader.prepare_sells_data()
loader.prepare_products_list()
df_lags, X, y = loader.prepare_training_data()

predictor = Predictor()
predictor.model_train(X, y)


@app.route("/", methods=["GET", "POST"])
def index():
    products = loader.productsList
    product = None
    message = ""
    history = []
    forecast = []
    forecast_text = ""

    if request.method == "POST":
        product = request.form.get("product_code", "").strip()
        if product not in products:
            message = "❌ Nieprawidłowy kod produktu lub brak danych."
        else:
            history, forecast, message = predictor.forecasting_www(product, df_lags, loader.le)

            # Konwersja typów do float - ważne dla JSON serializacji
            history = [float(x) for x in history] if history else []
            forecast = [float(x) for x in forecast] if forecast else []

            # Generujemy tekst z prognozą
            forecast_lines = [
                f"Prognozowana sprzedaż produktu {product} za miesiąc {i+1} od teraz: {pred:.2f}"
                for i, pred in enumerate(forecast)
            ]
            forecast_text = "\n".join(forecast_lines)

    return render_template(
        "index.html",
        products=products,
        product=product,
        message=message,
        history=history,
        forecast=forecast,
        forecast_text=forecast_text
    )


if __name__ == "__main__":
    app.run(debug=True)
