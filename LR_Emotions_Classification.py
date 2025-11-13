import joblib
import utilities
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

class LogisticRegressionClassifier:
    def __init__(self, ds):
        """
        Initialize parameters.
        """
        self.ds = ds
        self.label_names = None
        self.count_vectorizer = None
        self.mlb = None
        self.mlb_classifier = None
        self.train_counts = None
        self.dev_counts = None
        self.test_counts = None
        self.binary_labels_train = None
        self.binary_labels_dev = None
        self.binary_labels_test = None


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

            # Get emotion names
            self.label_names = train_set.features['labels'].feature.names

        except Exception as e:
            print(f"Error during loading data: {e}")
            raise
    
    def _extract_features(self):
        """
        Extract features and labels from the dataset.
        """
        try:
            # Initialize tokenizer and vectorizer
            # 28 emotions including neutral
            print("Initializing CountVectorizer with custom tokenizer...")
            self.count_vectorizer = CountVectorizer(analyzer=utilities.nltk_tokenizer)
            self.mlb = MultiLabelBinarizer(classes=range(28))

            print("Starting feature extraction...")
            self.train_counts = self.count_vectorizer.fit_transform(self.train_texts)
            self.dev_counts = self.count_vectorizer.transform(self.dev_texts)
            self.test_counts = self.count_vectorizer.transform(self.test_texts)

            # Convert GoEmotions labels to binary indicator matrix for comparison
            self.binary_labels_train = self.mlb.fit_transform(self.train_labels)
            self.binary_labels_dev = self.mlb.transform(self.dev_labels)
            self.binary_labels_test = self.mlb.transform(self.test_labels)

        except Exception as e:
            print(f"Error during feature extraction: {e}")
            raise


    def _initialize_classifier(self):
        """
        Initialize the multi-label classifier.
        """
        try:
            # Initialize base classifier using saga solver
            # saga should work well with large, sparse data and multi-label problems.
            # Wrap with OneVsRestClassifier using the strategy One-vs-the-rest (OvR) multiclass.
            base_lr = LogisticRegression(max_iter=1000, solver='saga', random_state=0)
            self.mlb_classifier = OneVsRestClassifier(base_lr)
        except Exception as e:
            priunt(f"Error initializing classifier: {e}")
            raise


    def train(self):
        """
        Train the classifier.
        """
        try:
            self._load_data()
            self._extract_features()
            self._initialize_classifier()
            print("Training classifier...")
            self.mlb_classifier.fit(self.train_counts, self.binary_labels_train)

        except Exception as e:
            print(f"Error during training: {e}")
            raise

def main():
    # Load the GoMotions datasets.
    ds = utilities.retrieve_dataset()

    # Initialize classifier and train it
    lr_classifier_obj = LogisticRegressionClassifier(ds)
    lr_classifier_obj.train()

    # Predict the class for each dev document. 
    lr_dev_predictions = lr_classifier_obj.mlb_classifier.predict(lr_classifier_obj.dev_counts)

    # Predict the class for each test document. 
    lr_test_predictions = lr_classifier_obj.mlb_classifier.predict(lr_classifier_obj.test_counts)

    # Print results
    print("Dev results:")
    utilities.print_results(lr_classifier_obj.binary_labels_dev, lr_dev_predictions, lr_classifier_obj.label_names)

    print()
    print("Test results:")
    utilities.print_results(lr_classifier_obj.binary_labels_test, lr_test_predictions, lr_classifier_obj.label_names)


if __name__ == "__main__":
    main()