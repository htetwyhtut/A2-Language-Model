import torch
import torch.nn as nn
import pickle

from flask import Flask, render_template, request
from class_function import LSTMLanguageModel, generate

# Importing training data
Data = pickle.load(open('../models/Data.pkl', 'rb'))

vocab_size = Data['vocab_size']
emb_dim = Data['emb_dim']
hid_dim = Data['hid_dim']
num_layers = Data['num_layers']
dropout_rate = Data['dropout_rate']
tokenizer = Data['tokenizer']
vocab = Data['vocab']

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define device
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model.load_state_dict(torch.load('../../best-val-lstm_lm_v2.pt', map_location=device))
model.eval()

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])

def index():
    # Home page
    if request.method == 'GET':
        return render_template('index.html', prompt='')
    
    # Page after user input
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        seq_len = int(request.form.get('seq'))
        temperature = 1.0
        seed = 0
        generation = generate(prompt, seq_len, temperature, model, tokenizer, 
                            vocab, device, seed)
        
        sentence = ' '.join(generation)
        return render_template('index.html', prompt=prompt, seq_len=seq_len, sentence=sentence)

if __name__ == '__main__':
    app.run(debug=True)
