import joblib
import numpy
import utilities
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report


class LogisticRegressionClassifier:
    def __init__(self, ds):
        # Store dataset
        self.ds = ds

        # Split datasets
        train_set = ds['train']
        dev_set = ds['validation']
        test_set = ds['test']

        # Extract texts and labels.
        # Not storing train texts to save space.
        train_texts, train_labels = train_set['text'], train_set['labels']
        self.dev_texts, self.dev_labels= dev_set['text'], dev_set['labels']
        self.test_texts, self.test_labels = test_set['text'], test_set['labels']

        # Get emotion names
        self.label_names = train_set.features['labels'].feature.names

        # Initialize tokenizer and vectorizer as attributes
        # 28 emotions including neutral
        print("Initializing CountVectorizer with custom tokenizer...")
        self.count_vectorizer = CountVectorizer(analyzer=utilities.nltk_tokenizer)
        self.mlb = MultiLabelBinarizer(classes=range(28))

        print("Starting feature extraction...")
        self.train_counts = self.count_vectorizer.fit_transform(train_texts)
        self.dev_counts = self.count_vectorizer.transform(self.dev_texts)
        self.test_counts = self.count_vectorizer.transform(self.test_texts)

        # Convert GoEmotions labels to binary indicator matrix for comparison
        self.binary_labels_train = self.mlb.fit_transform(train_labels)
        self.binary_labels_dev = self.mlb.transform(self.dev_labels)
        self.binary_labels_test = self.mlb.transform(self.test_labels)

        # Initialize classifier
        # Initialize base classifier using saga solver
        # saga should work well with large, sparse data and multi-label problems.
        # Wrap with OneVsRestClassifier using the strategy One-vs-the-rest (OvR) multiclass.
        base_lr = LogisticRegression(max_iter=1000, solver='saga', random_state=0)
        self.mlb_classifier = OneVsRestClassifier(base_lr)


    def train(self):
        # Train multi-label classifier
        self.mlb_classifier.fit(self.train_counts, self.binary_labels_train)


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
    print_results(lr_classifier_obj.binary_labels_dev, lr_dev_predictions, lr_classifier_obj.label_names)

    print()
    print("Test results:")
    print_results(lr_classifier_obj.binary_labels_test, lr_test_predictions, lr_classifier_obj.label_names)


if __name__ == "__main__":
    main()