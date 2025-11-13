import nltk
import re
import numpy
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# Tokenizer function using NLTK word_tokenize
def nltk_tokenizer(text):
    # Preprocess: removes [NAME] tokens 
    text = re.sub(r'\[NAME\]', '', text)  
    tokens = word_tokenize(text)
    return tokens


def retrieve_dataset():
    # Load full GoEmotions dataset
    return load_dataset("google-research-datasets/go_emotions")

# A helper function to print out macro averaged P, R, F, accuracy, and relaxed accuracy calculations
# Uses implementations of evaluation metrics from sklearn
def print_results(gold_labels, predicted_labels: numpy.ndarray, label_names):
    p,r,f,_ = precision_recall_fscore_support(gold_labels, 
                                            predicted_labels,
                                            average='macro',
                                            zero_division=0)
    acc = accuracy_score(gold_labels, predicted_labels)


    # Relaxed true positives method, check when predicted and gold labels match with at least one emotion.
    relaxed_tp_mask = numpy.any(numpy.logical_and(gold_labels, predicted_labels), axis=1)
    # Count how many were at least one true positive.
    relaxed_true_positives = numpy.sum(relaxed_tp_mask)
    # Accuracy calculation
    relaxed_accuracy = relaxed_true_positives / gold_labels.shape[0]

    print("Precision: ", p)
    print("Recall: ", r)
    print("F1: ", f)
    print("Accuracy: ", acc)
    print("Accuracy (relaxed, at-least one match):", relaxed_accuracy)
    print()

    # Detailed classification report
    print("Detailed Classification Report:")
    report = classification_report(
        gold_labels,
        predicted_labels,
        labels=range(len(label_names)),
        target_names=label_names,
        zero_division=0,
    )
    print(report)
    print()