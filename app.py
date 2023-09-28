from pathlib import Path

from flask import jsonify, Flask, request, Response
from joblib import load
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
MODEL_PATH = Path("models") / "trained_model.joblib"
model = load(MODEL_PATH)


app = Flask(__name__)


@app.route("/")
def health_check():
    return "Welcome. Everything is fine!"


# Prediction route
@app.route("/predict", methods=["POST"])
def predict() -> Response:
    data = request.json
    data = list(data.values())

    input_data = np.array(data).reshape(1, -1)

    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    predictions = model.predict(input_data)
    return jsonify({"predictions": predictions.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
