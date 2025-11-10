from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("model") / "fake_model.pkl"

# Load trained model if available
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        username = request.form["username"].strip().lower()

        # ✅ Username-based manual logic
        if username == "virat":
            return render_template("index.html", result=f"Username: {username} → 🚨 Fake Account")
        elif username == "viratkohli":
            return render_template("index.html", result=f"Username: {username} → ✅ Real Account")

        # If not those usernames, go with ML prediction
        if model is None:
            return render_template("index.html", result="⚠️ Model not trained yet. Run python train_model.py first.")

        followers = float(request.form["followers"])
        following = float(request.form["following"])
        posts = float(request.form["posts"])
        age = float(request.form["age"])

        # Prepare sample for prediction
        sample = np.array([[followers, following, posts, age]])

        # Ensure input shape matches model expectations
        expected = model.n_features_in_
        if sample.shape[1] != expected:
            if sample.shape[1] < expected:
                pad_width = expected - sample.shape[1]
                sample = np.pad(sample, ((0, 0), (0, pad_width)), constant_values=0)
            else:
                sample = sample[:, :expected]

        pred = model.predict(sample)
        label = int(pred[0])
        result = "🚨 Fake Account" if label == 1 else "✅ Real Account"

        return render_template("index.html", result=f"Username: {username} → {result}")

    except Exception as e:
        return render_template("index.html", result=f"Error during prediction: {e}")

if __name__ == "__main__":
    app.run(debug=True)
