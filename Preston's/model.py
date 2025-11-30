import pandas
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


seed = 1234





def training_split(path_name):
    document = pandas.read_csv(path_name)
    return train_test_split(
        document['comment_text'], 
        document[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
        test_size=0.2,
        random_state=seed
    )




def create_models(label_list, model_dict, x_tr, y_tr) -> None:
    for current_label in label_list:
        model = LinearSVC(tol=.0001, fit_intercept=True, C=0.75)
        model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
        model.fit(x_tr, y_tr[current_label])
        model_dict[current_label] = model




if __name__ == '__main__':
    
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    models = dict()

    X_tr, X_va, y_tr, y_va = training_split(labels)


    tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=60000, stop_words='english')

    X_train_tfidf = tfidf.fit_transform(X_tr)
    X_va_tfidf = tfidf.transform(X_va)

    create_models(labels, models, X_tr, y_tr)
    print('models created')

    for current_label in labels:
        print(f'{current_label}:')
        train_error = 1 - models[current_label].score(X_train_tfidf, y_tr[current_label])
        print(f'training error: {train_error}')
        val_error = 1 - models[current_label].score(X_va_tfidf, y_va[current_label])
        print(f'validation error: {val_error}')
        print()


    
