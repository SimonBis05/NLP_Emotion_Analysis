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

def print_multilabel_results(gold_labels, predicted_labels, label_names):
    """
    Prints evaluation metrics for multi-label classification (GoEmotions style).
    gold_labels: list of lists OR multi-hot numpy array
    predicted_labels: list of lists OR multi-hot numpy array
    label_names: list of emotion names
    """

    import numpy as np
    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
        classification_report
    )

    # If labels are lists of indices, convert to multi-hot
    if isinstance(gold_labels[0], list):
        num_labels = len(label_names)

        gold_mh = np.zeros((len(gold_labels), num_labels))
        pred_mh = np.zeros((len(predicted_labels), num_labels))

        for i, labs in enumerate(gold_labels):
            for lab in labs:
                gold_mh[i, lab] = 1

        for i, labs in enumerate(predicted_labels):
            for lab in labs:
                pred_mh[i, lab] = 1

        gold_labels = gold_mh
        predicted_labels = pred_mh

    # ----- Metrics -----
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    # Strict accuracy (all labels match exactly)
    strict_accuracy = accuracy_score(gold_labels, predicted_labels)

    # Relaxed accuracy (at least one correct label)
    relaxed_tp_mask = np.any(
        np.logical_and(gold_labels == 1, predicted_labels == 1),
        axis=1
    )
    relaxed_accuracy = np.sum(relaxed_tp_mask) / gold_labels.shape[0]

    # ----- Output -----
    print("===== Multi-Label Emotion Evaluation =====")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1 Score:  {f1:.4f}")
    print(f"Strict Accuracy: {strict_accuracy:.4f}")
    print(f"Relaxed Accuracy:{relaxed_accuracy:.4f}")
    print()

    print("----- Classification Report (Per Emotion) -----")
    print(classification_report(
        gold_labels,
        predicted_labels,
        target_names=label_names,
        zero_division=0
    ))
    print("================================================")
