from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Titanic model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        pclass = int(request.form.get('pclass'))
        sex = 1 if request.form.get('sex') == 'male' else 0
        age = float(request.form.get('age'))
        sibsp = int(request.form.get('sibsp'))
        parch = int(request.form.get('parch'))
        fare = float(request.form.get('fare'))
        embarked = request.form.get('embarked')
        
        # Encode embarked (assuming you use 'C', 'Q', 'S' as possible values)
        if embarked == 'C':
            embarked = 0
        elif embarked == 'Q':
            embarked = 1
        else:
            embarked = 2  # S
        
        # Create input feature vector
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        # Return the prediction result as a string (0 = not survived, 1 = survived)
        result = 'Survived' if prediction[0] == 1 else 'Did not survive'
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
