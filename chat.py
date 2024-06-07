import random
import json
import torch
from model import NeuralNet
from nltk_bot import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model data
with open('Intent.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"]), tag
    return "I'm sorry, I don't understand that request. Could you please rephrase it?", None

questions_dict = {}
for intent in intents['intents']:
    if 'questions' in intent and 'variables' in intent:
        questions_dict[intent['tag']] = list(zip(intent['questions'], intent['variables']))

def chatbot_logic(user_input, current_intent=None, responses=None):
    if responses is None:
        responses = {}

    print(f"Current Intent: {current_intent}, Responses: {responses}")

    if current_intent is None:
        response, tag = get_response(user_input)
        print(f"Response: {response}, Tag: {tag}")
        if tag in questions_dict:
            current_intent = tag
            responses = {}
            follow_up_question = questions_dict[tag][0][0]
            return f"{response}\n{follow_up_question}", current_intent
        else:
            return response, None
    else:
        variable = questions_dict[current_intent][len(responses)][1]
        print(f"Handling Variable: {variable}")
        try:
            if variable == "age":
                age = int(user_input)
                if age < 22:
                    current_intent = None
                    return "I'm sorry, but you need to be at least 22 years old to be eligible for a loan. Have a great day!", current_intent
                responses[variable] = age
            elif variable == "income":
                income = int(user_input)
                responses[variable] = income
            else:
                responses[variable] = user_input
        except ValueError:
            return f"Oops! It seems you've entered an invalid input for {variable}. Please enter a valid number.", current_intent

        if len(responses) < len(questions_dict[current_intent]):
            follow_up_question = questions_dict[current_intent][len(responses)][0]
            return follow_up_question, current_intent
        else:
            income = responses.get("income")
            if income is not None:
                if income < 5000:
                    loan_payout = income * 0.02
                else:
                    loan_payout = income * 0.03
                current_intent = None
                return f"Based on your income, you may qualify for a loan payout amount of ${loan_payout:.2f}. Do you want to continue with the loan application process? (yes/no)", current_intent
            else:
                current_intent = None
                return "Sorry, I couldn't determine your monthly income. Please try again later.", current_intent

# Ensure this module can be imported without executing main logic directly
if __name__ == "__main__":
    print("This script is intended to be imported as a module.")
