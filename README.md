# NLP: A2-Language-Model
Enjoy reading my A2 Assignment for NLP class.

## Author Info
Name: WIN MYINT@HTET WAI YAN HTUT (WILLIAM)
Student ID: st125326

## How to run the web app
1. Pull the github repository
2. Run
```sh
python app/app.py
```
3. Access the app using http://127.0.0.1:5000

## How to use website
1. Open a web browser and navigate to http://127.0.0.1:5000.
2. Enter a prompt and select a word limit.
2. Click "Generate Story" to see the result.

## Training Parameters
1. batch size            = 128
2. embedded dimension    = 1024
3. hidden dimension      = 1024
4. number of layers      = 2
5. dropout rate          = 0.65
6. learning rate         = 1e-3

## Screenshot of my web app
![Landing Page](<Landing Page.png>)
![Result Page](<Result Page.png>)

## Task 1: Training Data (1 points)
1. Corpus source - roneneldan/TinyStories
2. Data source: https://huggingface.co/datasets/roneneldan/TinyStories
3. Description: A Dataset containing synthetically generated (by GPT-3.5 and GPT-4) short stories that only use a small vocabulary.
4. Described in the following paper: https://arxiv.org/abs/2305.07759.
5. Number of rows: 2,141,709

## Task 2.1: Detail the steps taken to preprocess the text data. (1 points)
1. Load the Dataset
2. Define the Tokenizer
- Use `torchtext.data.utils.get_tokenizer` to create a tokenizer.<br><br>
3. Tokenize the Text
- Define a function to tokenize each example in the dataset.
- Use the `map` method to apply the tokenization function to the entire dataset.<br><br>
4. Remove Unnecessary Columns
- Remove the original `'text'` column after tokenization.<br><br>
5. Build Vocabulary and Numericalize Data
- Build a vocabulary using `torchtext.vocab.build_vocab_from_iterator`.
  - Set a minimum frequency threshold (e.g., `min_freq=3`) to filter out rare tokens.
- Add special tokens (e.g., `'<unk>'` and `'<eos>'`) if they don't already exist in the vocabulary. <br><br>
6. Inspect the Results
- Verify that the tokenization was applied correctly by inspecting the tokenized dataset.

## Task 2.2: Describe the model architecture and the training process. (1 points)

Model Architecture
The model is an LSTM-based Language Model implemented in PyTorch. It consists of the following key components:
1. Embedding Layer
2. **LSTM Layer
3. Dropout Layer
4. Fully Connected Layer
5. Initialization

Training Process (Data Preparation and Training Loop)
1. Tokenization and Numericalization
2. Batching
3. Training Loop
- Initialization,Forward Pass, Loss Calculation, Backpropagation, Evaluation, Checkpointing

Metrics (Perplexity):
  - Used to evaluate the model's performance.
  - Defined as exp(loss).
  - Lower perplexity indicates better performance.

Training Output
During training, the following metrics are printed for each epoch:
- Train Perplexity: Perplexity on the training set.
- Valid Perplexity: Perplexity on the validation set.

## Task 3. Text Generation - Web Application Development (2 points)
Provide documentation on how the web application interfaces with the language model.

1. Frontend (HTML/CSS)
2. Backend (Flask)
3. User Workflow

Step 1: User Input
The user visits the web application and sees a form with two fields:
Text Prompt: The user enters a starting sentence or phrase (e.g., "Once upon a time").
Max Word Limit: The user selects the maximum number of words to generate (options: 20, 50, 100).
The user clicks the "Generate Story" button to submit the form.

Step 2: Backend Processing
The Flask application (app.py) receives the form data via a POST request.
The application extracts the following from the form:
prompt: The text prompt entered by the user.
seq_len: The maximum word limit selected by the user.
The application calls the generate function from class_function.py with the following parameters:
prompt: The user's text prompt.
max_seq_len: The maximum word limit.
temperature: Controls the randomness of the generated text (default: 1.0).
model: The pre-trained LSTM language model.
tokenizer: Converts text into tokens for the model.
vocab: Maps tokens to indices and vice versa.
device: Specifies whether to use CPU or GPU (automatically detected).
seed: Ensures reproducibility (default: 0).

Step 3: Text Generation
The generate function processes the input prompt:
Tokenizes the prompt using the tokenizer.
Converts tokens into indices using the vocab.
The LSTM model generates text autoregressively:
The model predicts the next word based on the input sequence.
The process repeats until the maximum word limit is reached or an end-of-sequence token (<eos>) is generated.
The generated tokens are converted back into text using the vocab.

Step 4: Display Results
The generated text is returned to the Flask application.
The application renders the index.html template with the generated text and displays it to the user.
