import math
import utilities
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, ds):
        """
        Initialize parameters.
        """
        self.ds = ds
        self.train_texts = None
        self.dev_texts = None
        self.test_texts = None
        self.train_labels = None
        self.dev_labels = None
        self.test_labels = None
        self.train_primary_labels = None
        self.dev_primary_labels = None
        self.test_primary_labels = None
        self._load_data()
                
        # Tokenizer from utilities
        self.tokenize_fn = utilities.nltk_tokenizer

        # Initialize data structures
        self.classes = sorted(set(self.train_primary_labels))
        self.class_counts = {c: 0 for c in self.classes}
        self.word_counts = {c: defaultdict(int) for c in self.classes}
        self.total_words = {c: 0 for c in self.classes}
        self.vocab = set()
        
        # Track global frequency of all words across classes
        self.global_counts = defaultdict(int)

        for idx, (doc, label) in enumerate(zip(self.train_texts, self.train_primary_labels)):
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
                self.total_words[label] += 1
                self.global_counts[token] += 1
                
        self.vocab = sorted(list(self.vocab))

        MIN_FREQ = 3  # keep words appearing at least 3 times total

        filtered_vocab = [w for w in self.vocab if self.global_counts[w] >= MIN_FREQ]
        
        self.vocab = filtered_vocab
        self.vocab_size = len(self.vocab)
        
        # Compute class priors: log P(class)
        for c in self.classes:
            for w in list(self.word_counts[c].keys()):
                if w not in self.vocab:
                    # decrease total words count
                    self.total_words[c] -= self.word_counts[c][w]
                    del self.word_counts[c][w]

        self.num_docs = len(self.train_texts)
        self.num_classes = len(self.classes)

        # Compute class priors: log P(class)
        self.class_priors = {}
        for c in self.classes:
            prob = self.class_counts[c] / self.num_docs
            self.class_priors[c] = math.log(prob)

        # Compute smoothed likelihoods
        self.likelihoods = {c: {} for c in self.classes}

        for c in self.classes:
            for w in self.vocab:
                # Count of word w in class c
                count = self.word_counts[c].get(w, 0)

                # Laplace smoothing
                likelihood = (count + 1) / (self.total_words[c] + self.vocab_size)

                # Store log probability
                self.likelihoods[c][w] = math.log(likelihood)

    def _load_data(self):
        """
        Loading dataset and prepare it for training and testing.
        """
        try:
            # Split datasets
            train_set = self.ds['train']
            dev_set = self.ds['validation']
            test_set = self.ds['test']

            # Extract texts and labels.
            self.train_texts, self.train_labels = train_set['text'], train_set['labels']
            self.dev_texts, self.dev_labels= dev_set['text'], dev_set['labels']
            self.test_texts, self.test_labels = test_set['text'], test_set['labels']

            # Derive primary labels (first label) for training/evaluation compatibility
            self.train_primary_labels = [lst[0] if (isinstance(lst, list) and len(lst) > 0) else 0 for lst in self.train_labels]
            self.dev_primary_labels = [lst[0] if (isinstance(lst, list) and len(lst) > 0) else 0 for lst in self.dev_labels]
            self.test_primary_labels = [lst[0] if (isinstance(lst, list) and len(lst) > 0) else 0 for lst in self.test_labels]


        except Exception as e:
            print(f"Error during loading data: {e}")
            raise
        

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

