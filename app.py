from flask import Flask
from dataloader import DataLoader

app = Flask(__name__)

data = DataLoader


if __name__ == "__main__":
    app.run(debug=True)