import nltk
import re
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from datasets import load_dataset

# Tokenizer function using NLTK word_tokenize
def nltk_tokenizer(text):
    # Preprocess: removes [NAME] tokens 
    text = re.sub(r'\[NAME\]', '', text)  
    return word_tokenize(text)


def retrieve_dataset():

    # Load full GoEmotions dataset
    ds = load_dataset("google-research-datasets/go_emotions")

    # Example: how to access train, dev, and test datasets
    # train_data = ds['train']
    # val_data = ds['validation']
    # test_data = ds['test']

    # Example: how to access texts and labels from set
    # train_texts = train_data['text']
    # train_labels = train_data['labels']

    return ds
