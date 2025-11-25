# Overview

This project evaluates multiple machine learning and transformer-based models for emotion classification.  
Our implementation includes:

- **Naive Bayes Classifier**  
- **Logistic Regression Classifier**  
- **State-of-the-art BERT GoEmotions Baseline**  
- **Our fine-tuned BERT Emotion Classification Model**

---

# How to Run Our Code

## 1. Create a virtual environment (recommended)

```bash
python -m venv venv
```
## 2. Activate it
```bash
venv\Scripts\activate
```
## 3. Intsall virtual env requirements
```bash
pip intsall -r requirements.txt
```
## 4. Run Main function
```python
python ./main.py
```



# Project Structure
```
project/
│── bert_base_goemotion.py
│── bert_emotion_tunned_model.py
│── LR_Emotions_Classification.py
│── main.py
│── README.md
│── requirements.txt
│── utilities.py
```

# Modifying our Model
- To make changes to our model, locate the train_and_tune() function from the bert_emotion_tunned_model.py file and change training_args()
