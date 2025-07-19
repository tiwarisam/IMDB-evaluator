from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = joblib.load('horror_model.pkl')
scaler = joblib.load('horror_scaler.pkl')
label_encoders = joblib.load('horror_label_encoders.pkl')

def safe_encode(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form input
        popularity = float(request.form['popularity'])
        vote_count = int(request.form['vote_count'])
        budget = float(request.form['budget'])
        revenue = float(request.form['revenue'])
        runtime = float(request.form['runtime'])
        genre = request.form['genre_names']
        language = request.form['original_language']
        status = request.form['status']

        # Encode and scale input
        encoded = [
            popularity,
            vote_count,
            budget,
            revenue,
            runtime,
            safe_encode(genre, label_encoders['genre_names']),
            safe_encode(language, label_encoders['original_language']),
            safe_encode(status, label_encoders['status'])
        ]
        input_scaled = scaler.transform([encoded])
        predicted_rating = model.predict(input_scaled)[0]
        prediction = round(predicted_rating, 2)

    return render_template('index.html', prediction=prediction)
