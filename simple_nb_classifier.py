import math
import re
from collections import defaultdict
from utilities import retrieve_dataset, nltk_tokenizer 
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

class NaiveBayesClassifier:
    def __init__(self, documents, labels):

        # Tokenizer from utilities
        self.tokenize_fn = nltk_tokenizer
        
        # Initialize data structures
        self.classes = sorted(set(labels))
        self.class_counts = {c: 0 for c in self.classes}
        self.word_counts = {c: defaultdict(int) for c in self.classes}
        self.total_words = {c: 0 for c in self.classes}
        self.vocab = set()
        self.vocab_freq = defaultdict(int)
        
        print("Tokenizing documents and counting words...")
        
        for idx, (doc, label) in enumerate(zip(documents, labels)):
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1} documents...")
            
            # Tokenize document
            tokens = self._tokenize(doc)
            
            # Update class count
            self.class_counts[label] += 1
            
            # Count words
            for token in tokens:
                self.vocab.add(token)
                self.word_counts[label][token] += 1
                self.vocab_freq[token] += 1
                self.total_words[label] += 1
        
        print(f"Vocabulary size: {len(self.vocab)}")
        
        self.vocab = sorted(list(self.vocab))
        
        # Calculate basic statistics
        self.num_docs = len(documents)
        self.num_classes = len(self.classes)
        self.vocab_size = len(self.vocab)
        
        print(f"Classes: {self.classes}")
        print(f"Number of documents: {self.num_docs}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Vocabulary size: {self.vocab_size}")
        print()
        
        # Calculate class priors: P(class) = count(class) / total_documents
        print("Calculating class priors...")
        self.class_priors = {}
        for c in self.classes:
            prob = self.class_counts[c] / self.num_docs
            self.class_priors[c] = math.log(prob)
        
        # Calculate word likelihoods with Laplace smoothing
        self.likelihoods = {c: {} for c in self.classes}
        
        for c in self.classes:
            for w in self.vocab:
                # Count of word w in class c
                count = self.word_counts[c].get(w, 0)
                
                # Laplace smoothing: (count + 1) / (total_words + vocab_size)
                likelihood = (count + 1) / (self.total_words[c] + self.vocab_size)
                
                # Store as log probability
                self.likelihoods[c][w] = math.log(likelihood)
        

    def _tokenize(self, text):
        if not isinstance(text, str):
            return []

        # Use utilities tokenizer
        tokens = self.tokenize_fn(text)

        return tokens

    def sanity_check(self):
        # Check class priors
        prior_sum = sum(math.exp(p) for p in self.class_priors.values())
        assert abs(prior_sum - 1.0) < 0.01, f"Priors don't sum to 1: {prior_sum}"
        print(f"Class priors sum to 1.0")
        
        # Check that likelihoods are valid log-probabilities
        for c in self.classes:
            for w in self.vocab:
                assert isinstance(self.likelihoods[c][w], float), \
                    f"Likelihood for {c}, {w} is not a float"
                assert self.likelihoods[c][w] <= 0, \
                    f"Log likelihood should be <= 0: {self.likelihoods[c][w]}"
        
        print(f"All likelihoods are valid log-probabilities")
        print("Sanity checks passed!")
        print()

    def classify(self, test_instance):
        # Tokenize test instance
        tokens = self._tokenize(test_instance)
        
        # Calculate scores for each class
        scores = {}
        for c in self.classes:
            # Start with class prior: log P(class)
            score = self.class_priors[c]
            
            # Add likelihoods: sum of log P(word | class)
            for w in tokens:
                if w in self.likelihoods[c]:
                    score += self.likelihoods[c][w]
                else:
                    # For unseen words, use smoothed probability
                    unseen_prob = 1 / (self.total_words[c] + self.vocab_size)
                    score += math.log(unseen_prob)
            
            scores[c] = score
        
        # Return class with highest score
        predicted_class = max(scores, key=scores.get)
        return predicted_class

    def classify_batch(self, test_instances):

        predictions = []
        for instance in test_instances:
            pred = self.classify(instance)
            predictions.append(pred)
        return predictions

def load_goemotions_data(split='train', max_samples=None):

    print(f"Loading GoEmotions {split} dataset...")
    ds = retrieve_dataset()
    dataset = ds[split]

    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    # Extract texts and labels
    texts = dataset['text']
    labels_raw = dataset['labels']

    # Get emotion names
    label_names = dataset.features['labels'].feature.names
    
    # Keep multi-label lists as-is 
    labels = []
    for label_list in labels_raw:
        if isinstance(label_list, list):
            labels.append(list(label_list))
        elif label_list is None:
            labels.append([])
        else:
            # sometimes labels may already be a single int
            labels.append([label_list])
    
    print(f"Loaded {len(texts)} documents with {len(label_names)} emotion classes")
    # show distribution of primary labels for a quick sanity check
    primary_stats = [lst[0] if (isinstance(lst, list) and len(lst) > 0) else 0 for lst in labels]
    print(f"Label distribution (primary labels): {np.bincount(primary_stats, minlength=len(label_names))}")
    print()
    
    return texts, labels, label_names


def main():
    print()
    print("STEP 1: LOAD DATA")
    print("-" * 70)
    train_texts, train_labels, label_names = load_goemotions_data(
        split='train'
    )

    val_texts, val_labels, _ = load_goemotions_data(
        split='validation',
        max_samples=1000
    )

    # Derive primary labels (first label) for training/evaluation compatibility
    train_primary = [lst[0] if (isinstance(lst, list) and len(lst) > 0) else 0 for lst in train_labels]
    val_primary = [lst[0] if (isinstance(lst, list) and len(lst) > 0) else 0 for lst in val_labels]
    
    # Train classifier
    print("STEP 2: TRAIN MULTINOMIAL NAIVE BAYES")
    print("-" * 70)
    nb_classifier = NaiveBayesClassifier(
        documents=train_texts,
        labels=train_primary,
    )
    
    # Sanity check
    print("STEP 3: SANITY CHECKS")
    print("-" * 70)
    nb_classifier.sanity_check()
    
    # Evaluate on validation set
    print("STEP 4: EVALUATION")
    print("-" * 70)
    print("Making predictions on validation set...")
    val_predictions = nb_classifier.classify_batch(val_texts)
    
    accuracy_primary = accuracy_score(val_primary, val_predictions)
    f1_macro_primary = f1_score(val_primary, val_predictions, average='macro', zero_division=0)
    f1_weighted_primary = f1_score(val_primary, val_predictions, average='weighted', zero_division=0)

    print(f"Accuracy (primary-label compare): {accuracy_primary:.4f}")
    print(f"F1-Score (Macro, primary): {f1_macro_primary:.4f}")
    print(f"F1-Score (Weighted, primary): {f1_weighted_primary:.4f}")
    print()

    # Multi-label-aware metrics: prediction is correct if it matches any true label
    correct_any = sum(1 for pred, truths in zip(val_predictions, val_labels) if pred in truths)
    accuracy_any = correct_any / len(val_predictions) if len(val_predictions) > 0 else 0.0

    print(f"Accuracy (any-label match): {accuracy_any:.4f}")
    print()
    
    # Detailed classification report
    print("Detailed Classification Report:")
    report = classification_report(
        val_primary,
        val_predictions,
        labels=range(len(label_names)),
        target_names=label_names,
        zero_division=0,
    )
    print(report)
    print()


if __name__ == "__main__":
    main()
