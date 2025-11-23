import nltk
import re
import numpy
import torch
nltk.download('punkt_tab')
from datasets import Sequence, Features, Value, DatasetDict
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score


# Emotion labels
emotions = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]
# Number of emotions within the dataset
num_labels = 28

def bert_data_processor(dataset, tokenizer):
    """ Preprocess function to process the dataset and convert it into proper sets for training and evaluation for bert models"""

    # Function to convert labels from lists to tensors
    def convert_labels(data):
        global num_labels

        # Tokenize the text
        result = tokenizer(
            data["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        labels = []
        for label_list in data['labels']:
            multi_hot = [0.0] * num_labels
            for label in label_list:
                multi_hot[label] = 1.0
            labels.append(multi_hot)
        
        result["labels"] = labels
        return result

    # Define data type for each feature
    features = Features({
        'input_ids': Sequence(Value('int32')),
        'attention_mask': Sequence(Value('int32')),
        'labels': Sequence(Value('float32'))
    })

    # Tokenize datasets and convert labels
    tokenized_datasets = dataset.map(convert_labels, batched=True, remove_columns=dataset["train"].column_names, features = features)
    # Set dataset format for PyTorch
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=False)
    return tokenized_datasets

# Tokenizer function using NLTK word_tokenize
def nltk_tokenizer(text):
    """ Standard tokenizer for LR and NB models. """
    # Preprocess: removes [NAME] tokens 
    text = re.sub(r'\[NAME\]', '', text)
    tokens = word_tokenize(text)
    return tokens


def retrieve_dataset():
    # Load full GoEmotions dataset
    return load_dataset("google-research-datasets/go_emotions")

def retrieve_dataset_50():
    # Load 50% of train split
    train_50 = load_dataset("google-research-datasets/go_emotions", split="train[:50%]")

    # Load full test and validation splits
    test = load_dataset("google-research-datasets/go_emotions", split="test")
    validation = load_dataset("google-research-datasets/go_emotions", split="validation")

    # Combine into DatasetDict obj
    dataset_50 = DatasetDict({
        "train": train_50,
        "test": test,
        "validation": validation
    })
    return dataset_50


# A helper function to print out evaluation metrics for NB 
def print_results_nb(gold_labels, gold_labels_primary, predicted_labels):
        
    # Multi-label-aware metrics: prediction is correct if it matches any true label
    correct_any = sum(1 for pred, truths in zip(predicted_labels, gold_labels) if pred in truths)
    relaxed_accuracy = correct_any / len(predicted_labels) if len(predicted_labels) > 0 else 0.0
    
    return print_results(gold_labels_primary, predicted_labels, relaxed_accuracy)



# A helper function to print out evaluation metrics for LR
def print_results_lr(gold_labels, predicted_labels):

    # Relaxed true positives method, check when predicted and gold labels match with at least one emotion.
    relaxed_tp_mask = numpy.any(numpy.logical_and(gold_labels, predicted_labels), axis=1)
    # Count how many were at least one true positive.
    relaxed_true_positives = numpy.sum(relaxed_tp_mask)
    # Accuracy calculation
    relaxed_accuracy = relaxed_true_positives / gold_labels.shape[0]

    # Subset accuracy (exact match)
    subset_accuracy = accuracy_score(gold_labels, predicted_labels)

    return print_results(gold_labels, predicted_labels, relaxed_accuracy, subset_accuracy)



# Uses implementations of evaluation metrics from sklearn
def print_results(gold_labels, predicted_labels, relaxed_accuracy=0, subset_accuracy=0):

    # Calculate metrics
    f1_micro = f1_score(gold_labels, predicted_labels, average='micro')
    f1_macro = f1_score(gold_labels, predicted_labels, average='macro')
    f1_weighted = f1_score(gold_labels, predicted_labels, average='weighted')

    print("F1 Macro: ", f1_macro)
    print("F1 Micro: ", f1_micro)
    print("F1 Weighted: ", f1_weighted)
    print("Accuracy (relaxed, at-least one match):", relaxed_accuracy)
    if subset_accuracy != None:
        print("Subset Accuracy (exact match):", subset_accuracy)

    # Detailed classification report
    report = classification_report(
        gold_labels,
        predicted_labels,
        labels=range(len(emotions)),
        target_names=emotions,
        zero_division=0,
    )
    return report



def compute_metrics(eval_pred):
    """Evaluation function for bert models"""
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    
    # Convert probabilities to binary predictions (threshold at 0.5)
    predicted_labels = (probs.numpy() > 0.5).astype(int)
    gold_labels = labels.astype(int)
    
    # Calculate metrics
    f1_micro = f1_score(gold_labels, predicted_labels, average='micro')
    f1_macro = f1_score(gold_labels, predicted_labels, average='macro')
    f1_weighted = f1_score(gold_labels, predicted_labels, average='weighted')
    
    # Subset accuracy (exact match)
    subset_accuracy = accuracy_score(gold_labels, predicted_labels)
    
    # Relaxed true positives method, check when predicted and gold labels match with at least one emotion.
    relaxed_tp_mask = numpy.any(numpy.logical_and(gold_labels, predicted_labels), axis=1)
    # Count how many were at least one true positive.
    relaxed_true_positives = numpy.sum(relaxed_tp_mask)
    # Accuracy calculation
    relaxed_accuracy = relaxed_true_positives / gold_labels.shape[0]
    
    return {
        'f1_macro': f1_micro,
        'f1_macro': f1_macro, # Optimal value for comparison
        'f1_weighted': f1_weighted,
        'Subset Accuracy (exact match)': subset_accuracy,
        'Accuracy (relaxed, at-least one match)': relaxed_accuracy
    }


def predict_batch(bert_tokenizer, model, top_n=5):
    """Predict emotions for multiple texts bert model"""
    # Make manual predictions
    texts = [
        "I'm so excited about this opportunity!",
        "This makes me really angry and frustrated.",
        "I'm grateful for all your help and support.",
        "Who is this Wild team? Where have they been?"
    ]

    # Move model to CPU
    model = model.to('cpu')

    # Tokenize all texts
    inputs = bert_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Move inputs to CPU
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)
    
    # Process each text's predictions
    all_predictions = []
    for text_probs in probabilities:
        # Get all emotions with their scores
        predictions = []
        for idx, prob in enumerate(text_probs):
            predictions.append({
                'emotion': emotions[idx],
                'score': float(prob)
            })
        
        # Sort by score and take top N
        predictions.sort(key=lambda x: x['score'], reverse=True)
        top_predictions = predictions[:top_n]
        
        all_predictions.append(top_predictions)

    for text, preds in zip(texts, all_predictions):
        print(f"\nText: '{text}'")
        print("Emotions:")
        if preds:
            for pred in preds:
                print(f"  {pred['emotion']:<20} {pred['score']:.4f}")
        else:
            print("  No emotions detected above threshold")
