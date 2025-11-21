import torch
import utilities
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Base model - DistilRoBERTa
original_model_name = "distilbert/distilroberta-base"
# Our fine tuned DistilRoBERTa model
trained_model_path = "./best_tuned_model"


class BertTunedGoMotion:
    def __init__(self, ds):
        global original_model_name, trained_model_path

        # Detect device
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
        # Load the model tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(original_model_name)

        # Load our tuned DistilRoBERTa model.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            original_model_name,
            num_labels=utilities.num_labels,
            problem_type="multi_label_classification"
        )

        # Set to evaluation mode
        self.model.eval()  

        # Preprocess the datasets using the tokenizer
        self.processed_ds = utilities.bert_data_processor(ds, tokenizer=self.tokenizer)
        self.trainer = None


    def load(self):
        try:
            # Set minimal training arg for evaluation
            training_args = TrainingArguments(
                output_dir="./trainer_output",
                per_device_eval_batch_size=16,
                fp16=torch.cuda.is_available(),  # FP16 only if GPU exists
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

    def train_and_tune(self):
        """
        Train and fine tune the model.
        """
        try:            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir="./distilroberta-goemotions-tuned", # Directory where model checkpoints and outputs will be saved
                eval_strategy="epoch", # Run evaluation at the end of each training epoch
                save_strategy="epoch", # Save model checkpoint at the end of each epoch
                per_device_train_batch_size=16, # Number of training samples processed per device in each batch
                gradient_accumulation_steps=2, # Accumulate gradients over 2 steps before updating weights (effective batch size = 16 * 2 = 32)
                learning_rate=5e-5, # Initial learning rate for the optimizer (0.00005)
                num_train_epochs=4, # Total number of complete passes through the training dataset
                weight_decay=0.01, # L2 regularization penalty to prevent overfitting
                warmup_ratio=0.1, # Proportion of training steps (10%) to gradually increase learning rate from 0 to the target rate
                load_best_model_at_end=True,  # Load the best performing checkpoint at the end of training
                metric_for_best_model="f1_macro", # Use macro-averaged F1 score to determine the best model
                greater_is_better=True, # Higher F1 score indicates better model performance
                save_total_limit=2, # Keep only the 2 most recent checkpoints to save space
                fp16=True, # Enable mixed precision training (16-bit floating point) for faster training and reduced memory usage
                logging_steps=100, # Log training metrics every 100 steps
            )

            # Create Trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.processed_ds['train'],
                eval_dataset=self.processed_ds['validation'],
                compute_metrics=utilities.compute_metrics,
            )

            # Train the model
            self.trainer.train()

        except Exception as e:
            print(f"Error during training: {e}")
            raise

