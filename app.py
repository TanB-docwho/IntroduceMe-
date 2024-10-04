from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

def call_api(endpoint):
    response = requests.get(endpoint)
    response.raise_for_status()  # Raise an exception for HTTP errors
    api_data = response.json()
    
    # Example formatting: Assuming the API returns a list of items
    formatted_response = "\n".join([f"- {item}" for item in api_data])
    return formatted_response

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input', '')
    
    # Example: Call the API internally to enhance the response
    try:
        api_response = call_api("https://api.example.com/data")
        response = f"Chatbot: {user_input}\nAdditional Information: {api_response}"
    except requests.exceptions.RequestException as e:
        response = f"Chatbot: {user_input}\nError calling API: {e}"
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)