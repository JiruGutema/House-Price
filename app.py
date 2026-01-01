import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
MODEL_PATH = "house_price_model_bundle/model.joblib"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            values = [float(request.form[f]) for f in FEATURES]

            x = np.array(values).reshape(1, -1)
            prediction = float(model.predict(x)[0])
            prediction = f"${prediction:,.2f}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.json
    x = np.array([data[f] for f in FEATURES]).reshape(1, -1)
    price = float(model.predict(x)[0])

    return jsonify({"predicted_house_price": price})


@app.route("/health")
def health():
    return {"status": "ok"}




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
