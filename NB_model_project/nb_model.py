import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, classification_report


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[^a-z0-9@#%*!?$\-_/ ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train["clean_text"] = train["comment_text"].apply(clean_text)
test["clean_text"] = test["comment_text"].apply(clean_text)


tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(4,6),
    max_features=50000
)

X = tfidf.fit_transform(train["clean_text"])
X_test = tfidf.transform(test["clean_text"])
y = train[labels]



#Split training and validation

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

naive_bayes = OneVsRestClassifier(MultinomialNB(alpha=0.1))
naive_bayes.fit(X_train, y_train)

y_val_pred_proba = naive_bayes.predict_proba(X_val)
y_val_pred = (y_val_pred_proba > 0.5).astype(int)

print("CLASSIFICATION REPORT (Validation Split)")
print(classification_report(y_val, y_val_pred, 
                            target_names=labels, 
                            zero_division = 0))

auc = roc_auc_score(y_val, y_val_pred_proba, average="macro")
print(f"Validation ROC-AUC: {auc:.4f}")