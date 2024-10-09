from flask import session, Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask_cors import CORS
from functions import calculate_bmi, create_workout_plan_with_BMI, create_diet_plan_with_BMI, create_workout_plan, create_diet_plan

app = Flask(__name__)
CORS(app)
app.secret_key = "eaf8b1d1a73c4097a53baec1ab5c1d41"

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents, words, classes, and the trained model
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", 'rb'))
classes = pickle.load(open("classes.pkl", 'rb'))
model = load_model("chatbot_model.h5")


# Tokenizes and lemmatizes a sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Converts a sentence into a bag of words (vectorized form)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Predict the intent class based on the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    print(f"BOW Shape: {np.array([bow]).shape}")  # Debugging the shape
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


# Selects a random response based on predicted intent
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I didn't understand that. Can you try again?"


@app.route("/")
def index():
    session["bmi_category"] = " "
    session["order"] = ""
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        if "msg" not in request.form:
            return jsonify({"error": "No message provided"}), 400

        msg = request.form["msg"]
        input = msg.strip()  # Sanitize input

        # Predict intent
        ints = predict_class(input)
        print(f"User Input: {input}")
        print(f"Predicted Intent: {ints}")

        # Access session variables
        print(f"BMI Category: {session.get('bmi_category', 'Not Set')}")
        print(f"Order: {session.get('order', 'Not Set')}")

        if ints and ints[0]['intent'] in ['personalized_workout_plan', 'personalized_diet_plan']:
            session["order"] = ints[0]['intent']
            print(f"Session Order: {session['order']}")

        # Handle BMI and workout/diet plans
        if "," in input:
            try:
                weight, height, age = map(float, input.split(','))
                result = calculate_bmi(weight, height, age)
                response = result[0]  # Get the response only
                session["bmi_category"] = result[1]  # Store bmi_category in session
                return response
            except ValueError:
                return "Please enter valid numbers for weight, height, and age."

        if input.lower() in ["beginner", "intermediate", "advanced"]:
            print(f"User level: {input.lower()}")

            if session.get("order") == 'personalized_workout_plan':
                if session.get("bmi_category") and session["bmi_category"] != " ":
                    response = create_workout_plan_with_BMI(input.lower(), session["bmi_category"])
                else:
                    response = create_workout_plan(input.lower())
                return response

            elif session.get("order") == 'personalized_diet_plan':
                if session.get("bmi_category") and session["bmi_category"] != " ":
                    response = create_diet_plan_with_BMI(input.lower(), session["bmi_category"])
                else:
                    response = create_diet_plan(input.lower())
                return response

        response = get_response(ints, intents)
        return response
    except Exception as e:
        # Log the error and return a message
        print(f"Error: {str(e)}")
        return jsonify({"error": "An error occurred"}), 500

    print("Inside Chat function")
    msg = request.form["msg"]
    input = msg.strip()  # Sanitize input

    # Predict intent
    ints = predict_class(input)
    print(f"User Input: {input}")
    print(f"Predicted Intent: {ints}")

    # Access session variables
    print(f"BMI Category: {session.get('bmi_category', 'Not Set')}")
    print(f"Order: {session.get('order', 'Not Set')}")

    if ints and ints[0]['intent'] in ['personalized_workout_plan', 'personalized_diet_plan']:
        session["order"] = ints[0]['intent']
        print(f"Session Order: {session['order']}")

    # If user enters values for BMI calculation
    if "," in input:
        try:
            weight, height, age = map(float, input.split(','))
            result = calculate_bmi(weight, height, age)
            response = result[0]  # Get the response only
            session["bmi_category"] = result[1]  # Store bmi_category in session
            return response
        except ValueError:
            return "Please enter valid numbers for weight, height, and age."

    # If user specifies workout/diet level
    if input.lower() in ["beginner", "intermediate", "advanced"]:
        print(f"User level: {input.lower()}")

        if session.get("order") == 'personalized_workout_plan':
            print("Processing personalized workout plan")
            if session.get("bmi_category") and session["bmi_category"] != " ":
                response = create_workout_plan_with_BMI(input.lower(), session["bmi_category"])
            else:
                response = create_workout_plan(input.lower())
            return response

        elif session.get("order") == 'personalized_diet_plan':
            print("Processing personalized diet plan")
            if session.get("bmi_category") and session["bmi_category"] != " ":
                response = create_diet_plan_with_BMI(input.lower(), session["bmi_category"])
            else:
                response = create_diet_plan(input.lower())
            return response

    # Default response
    response = get_response(ints, intents)
    return response


if __name__ == '__main__':
    app.run(debug=True)
