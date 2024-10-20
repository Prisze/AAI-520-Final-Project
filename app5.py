from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
import argparse
import os

# Initialize Flask app
app = Flask(__name__)

# Set device to CUDA if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Argument parser to get model path from user
parser = argparse.ArgumentParser(description='Run the chatbot with specified model path.')
parser.add_argument('--model_path', type=str, default=None, help='Path to the model directory')
args = parser.parse_args()

# If model_path is not provided, use a default relative path or fallback
model_path = args.model_path if args.model_path else os.getenv('MODEL_PATH', 'models/default_model')

# Load the pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')
tokenizer.pad_token = tokenizer.eos_token

class Chat:
    def __init__(self, chatbot, max_response_len=12):
        self.chatbot = chatbot
        self.messages = []
        self.max_response_len = max_response_len
        self.temperature = 0.4  # Default temperature
        self.top_p = 0.6  # Default top_p

    def send(self, text: str):
        self.messages.append(f"User: {text}")
        prompt = '<|endoftext|>'.join(self.messages) + '<|endoftext|>'
        max_tokens = 256
        prompt_tokens = self.chatbot.tokenizer.encode(prompt)
        if len(prompt_tokens) > max_tokens:
            prompt_tokens = prompt_tokens[-max_tokens:]
            prompt = self.chatbot.tokenizer.decode(prompt_tokens)
        response = self.chatbot.generate(prompt, max_length=self.max_response_len, temperature=self.temperature, top_p=self.top_p)
        cleaned_response = re.sub(r"User: |Bot: ", "", response)
        self.messages.append(f"Bot: {cleaned_response}")
        return cleaned_response

    def reset_conversation(self):
        self.messages = []

    def set_mood(self, temperature: float, top_p: float):
        self.temperature = temperature
        self.top_p = top_p

class ChatBot:
    def __init__(self, model_path: str, device=None):
        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, text: str, max_length: int = None, temperature: float = 0.3, top_p: float = 0.7) -> str:
        if not max_length:
            max_length = min(len(self.tokenizer.encode(text)) // 2, 100)  
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                max_new_tokens=max_length,
                no_repeat_ngram_size=5,
                early_stopping=True,
                num_beams=2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.5,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                length_penalty=0.3
            )
            response_outputs = outputs[:, len(inputs['input_ids'][0]):]
            response = self.tokenizer.batch_decode(response_outputs, skip_special_tokens=True)[0]
            return response

    def create_chat(self) -> Chat:
        return Chat(self)

# Load the ChatBot model
chatBot = ChatBot(model_path)
conversation = chatBot.create_chat()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    response = conversation.send(user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
