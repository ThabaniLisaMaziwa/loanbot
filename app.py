from flask import Flask, render_template, request, jsonify
from chat import chatbot_logic

app = Flask(__name__)

@app.route('/')
def index_get():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print(f"Received data: {data}")
    text = data.get("message")
    current_intent = data.get("current_intent")
    responses = data.get("responses", {})

    response, new_intent = chatbot_logic(text, current_intent, responses)
    message = {"answer": response, "intent_state": new_intent}
    print(f"Response: {message}")
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)
