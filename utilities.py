import nltk
import re
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from datasets import load_dataset

# Tokenizer function using NLTK word_tokenize
def nltk_tokenizer(text):
    # Preprocess: removes [NAME] tokens 
    text = re.sub(r'\[NAME\]', '', text)  
    tokens = word_tokenize(text)
    return tokens


def retrieve_dataset():
    # Load full GoEmotions dataset
    return load_dataset("google-research-datasets/go_emotions")

