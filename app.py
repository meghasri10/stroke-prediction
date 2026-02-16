from flask import redirect, url_for
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("stroke_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form.get('age', 0))
        hypertension = float(request.form['hypertension'])
        heart_disease = float(request.form['heart_disease'])
        glucose = float(request.form.get('glucose', 0))
        bmi = float(request.form.get('bmi', 0))
        smoking_map = {
            "never smoked": 0,
            "formerly smoked": 1,
            "smokes": 2,
            "Unknown": 3
        }

        smoking_status = smoking_map.get(request.form['smoking_status'], 0)

        final_input = np.array([[age, hypertension, heart_disease,
                                 glucose, bmi, smoking_status]])

        prediction = model.predict(final_input)[0]

        if prediction == 1:
            result_text = "⚠ High risk of stroke!"
        else:
            result_text = "✅ Low risk of stroke."

        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"⚠ Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
