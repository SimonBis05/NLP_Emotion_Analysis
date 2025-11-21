
import utilities
import LR_Emotions_Classification as lr
import simple_nb_classifier as nb
import bert_base_go_emotion as base
import bert_emotion_tuned_model as tuned


def baseline_nb(ds):
    # Load data 
    nb_classifier_obj = nb.NaiveBayesClassifier(ds)
    # Sanity check
    nb_classifier_obj.sanity_check()

    # Predict the class for each dev document. 
    nb_dev_predictions = nb_classifier_obj.classify_batch(nb_classifier_obj.dev_texts)

    # Predict the class for each test document. 
    nb_test_predictions = nb_classifier_obj.classify_batch(nb_classifier_obj.test_texts)

    # Print results
    print("Dev results:")
    utilities.print_results_nb(nb_classifier_obj.dev_labels, nb_classifier_obj.dev_primary_labels, nb_dev_predictions)


    print()
    print("Test results:")

    report = utilities.print_results_nb(nb_classifier_obj.test_labels, nb_classifier_obj.test_primary_labels, nb_test_predictions)
    # print("Detailed Classification Report:")
    # print(report)
    # print()


def baseline_lr(ds):
    """ Logistic regression classifier perform as a baseline"""
    # Initialize classifier and train it
    lr_classifier_obj = lr.LogisticRegressionClassifier(ds)
    lr_classifier_obj.train()

    # Predict the class for each dev document. 
    lr_dev_predictions = lr_classifier_obj.mlb_classifier.predict(lr_classifier_obj.dev_counts)

    # Predict the class for each test document. 
    lr_test_predictions = lr_classifier_obj.mlb_classifier.predict(lr_classifier_obj.test_counts)

    # Print results
    print("Dev results:")
    utilities.print_results_lr(lr_classifier_obj.binary_labels_dev, lr_dev_predictions)

    print()
    print("Test results:")
    
    report = utilities.print_results_lr(lr_classifier_obj.binary_labels_test, lr_test_predictions)
    # print("Detailed Classification Report:")
    # print(report)
    # print()


def bert_model_pretrained(ds):
    """ bhadresh-savani/bert-base-go-emotion model perform as the state of the art model """
    # Initialize classifier
    origBertModel = base.BertBaseGoMotion(ds)
    origBertModel.load()

    # Evaluate on test set
    test_results = origBertModel.trainer.evaluate(origBertModel.processed_ds["test"])
    print(test_results)

    # Evaluate on manual texts
    batch_predictions = utilities.predict_batch(origBertModel.tokenizer, origBertModel.model)


def bert_model_tuned(ds):
    """ our lightweight efficient DistilRoBERTa model that we tuned for GoEmotions dataset """ 
    # Initialize classifier
    tunedBert = tuned.BertTunedGoMotion(ds)

    #Load final model for evaluation
    tunedBert.train_and_tune()

    # Evaluate on test set
    test_results = tunedBert.trainer.evaluate(tunedBert.processed_ds["test"])
    print(test_results)

    # Evaluate on manual texts
    batch_predictions = utilities.predict_batch(tunedBert.tokenizer, tunedBert.model)


def main():
    # Load the GoMotions datasets.
    ds = utilities.retrieve_dataset()
    reduced_ds = utilities.retrieve_dataset_50()

    print("")
    print("Baseline Naive Base:")
    print("-" * 60)
    baseline_nb(ds)
    print("")

    print("Baseline Logistic Regression:")
    print("-" * 60)
    baseline_lr(ds)
    print("")

    print("Competitor State-of-the-art Bert model for GoMotions:")
    print("-" * 60)
    bert_model_pretrained(ds)
    print("")

    print("Our own trained tuned Bert model 50:")
    print("-" * 60)
    bert_model_tuned(reduced_ds)
    # bert_model_tuned(ds)




if __name__ == "__main__":
    main()




