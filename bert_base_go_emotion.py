import utilities
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

original_model_name = "bhadresh-savani/bert-base-go-emotion"

class BertBaseGoMotion:
    def __init__(self, ds):

        # Load the model tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(original_model_name)

        # Load bhadresh-savani/bert-base-go-emotion model.
        self.model = AutoModelForSequenceClassification.from_pretrained(original_model_name)

        # Set to evaluation mode
        self.model.eval() 

        # Preprocess the datasets using bert tokenizer
        self.processed_ds = utilities.bert_data_processor(ds, tokenizer=self.tokenizer)
        self.trainer = None

    def load(self):
        try:
           # Set minimal training arg for evaluation
            training_args = TrainingArguments(
                output_dir="./trainer_output",
                per_device_eval_batch_size=16
            )

           # Create Trainer for evaluation
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                compute_metrics=utilities.compute_metrics
            )
        except Exception as e:
            print(f"Error during loading model: {e}")
            raise