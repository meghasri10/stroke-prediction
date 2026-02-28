from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)

# Load model
model = pickle.load(open("stroke_model.pkl", "rb"))

# Create database
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    age REAL,
                    hypertension REAL,
                    heart_disease REAL,
                    glucose REAL,
                    bmi REAL,
                    smoking_status REAL,
                    prediction TEXT
                )''')
    conn.commit()
    conn.close()
init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Get form data
    age = float(request.form["age"])
    hypertension = int(request.form["hypertension"])
    heart_disease = int(request.form["heart_disease"])
    glucose = float(request.form["avg_glucose_level"])
    bmi = float(request.form["bmi"])
    smoking = int(request.form["smoking_status"])

    # 2. Prepare data for model
    input_data = np.array([[age, hypertension, heart_disease, glucose, bmi, smoking]])
    print("Input data:", input_data)

    # 3. Make prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    print("Prediction:", prediction)
    print("Probability:", prob)

    # 4. Decide result
    result = "High Risk" if prediction == 1 else "Low Risk"

    # 5. Save to database
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("""
           INSERT INTO predictions 
           (age, hypertension, heart_disease, glucose, bmi, smoking_status, prediction)
           VALUES (?, ?, ?, ?, ?, ?, ?)
       """, (age, hypertension, heart_disease, glucose, bmi, smoking, result))
    conn.commit()
    conn.close()

    # 6. Return result
    return render_template("result.html", result=result, probability=round(prob * 100, 2))


@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions")
    data = c.fetchall()
    conn.close()
    return render_template("dashboard.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)
