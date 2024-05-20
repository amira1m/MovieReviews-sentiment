import glob
import pandas as pd
import os
import re
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


base_path = 'txt_sentoken'

def read_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# Get all text files in the 'pos' and 'neg' folders
pos_files = glob.glob(os.path.join(base_path, 'pos', '*.txt'))
neg_files = glob.glob(os.path.join(base_path, 'neg', '*.txt'))

data = []
# Read positive files and assign the label 'positive'
for filepath in pos_files:
    text = read_text_file(filepath)
    data.append({'text': text, 'label': 'positive'})

# Read negative files and assign the label 'negative'
for filepath in neg_files:
    text = read_text_file(filepath)
    data.append({'text': text, 'label': 'negative'})

df = pd.DataFrame(data)
X =df['text'].astype(str)
Y = df['label'].astype(str)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=53 )

def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)

def convert_to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def tokenize(text):
    return word_tokenize(text)

def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def stem(tokens):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in tokens]

def join_tokens(tokens):
    return ' '.join(tokens)

# Function to process the data
def preprocessing(X_train):
    X_train = X_train.apply(remove_html_tags)
    X_train = X_train.apply(convert_to_lowercase)
    X_train = X_train.apply(remove_punctuation)
    X_train = X_train.apply(tokenize)
    X_train = X_train.apply(remove_stop_words)
    X_train = X_train.apply(stem)
    X_train = X_train.apply(join_tokens)
    return X_train

def vectorizer(X_train ,X_test):
    tfidf = TfidfVectorizer(max_features=2500)
    X_train = tfidf.fit_transform(X_train)
    # Fit on the reconstructed strings
    # model_filename = "tf_idf_vectorizer.pkl"
    # joblib.dump(tfidf, model_filename)
    X_test = tfidf.transform(X_test)  # Transform on the reconstructed strings
    return X_train,X_test

def scaling(X_train , X_test ) :
    scaler = StandardScaler(with_mean=False)
    # Fit and transform the training data
    X_train = scaler.fit_transform(X_train)  # Standardize features to mean=0, std=1
    # model_filename = "standard_scaler.pkl"
    # joblib.dump(scaler, model_filename)
    # Transform the test data
    X_test = scaler.transform(X_test)
    return X_train, X_test


def reduce_dimentions(X_train , X_test):
    svd = TruncatedSVD(n_components=200)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
    return X_train_reduced, X_test_reduced

def label_encode(Y_train,Y_test):
    label_encoder = LabelEncoder()
    Y_train = label_encoder.fit_transform(Y_train)
    # model_filename = "label_encoder.pkl"
    # joblib.dump(label_encoder, model_filename)
    Y_test= label_encoder.transform(Y_test)
    return Y_train, Y_test

def maxabs_scaler(X_train ,X_test):
    scaler = MaxAbsScaler()
    # Fit and transform the training data
    X_train = scaler.fit_transform(X_train)
    # model_filename = "maxabs_scaler.pkl"
    # joblib.dump(scaler, model_filename)
    X_test= scaler.transform(X_test)
    return X_train , X_test

def classify_svm (X_train, Y_train, X_test, Y_test):
    param_grid = {'C': [0.01,.1, 1]
                  ,'gamma':[0.2,0.3,0.5]}
    grid_search = GridSearchCV(SVC(kernel='linear', probability=True) ,param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)
    Y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(Y_train, Y_train_pred) * 100
    print("Training Accuracy:", train_accuracy)
    Y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100
    print("Best model accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))
    # model_filename = "svm.pkl"
    # joblib.dump(best_model, model_filename)
    return best_model,train_accuracy, accuracy


def classify_logistic(X_train, Y_train, X_test, Y_test):
    param_grid = {
    'C': [.001, .01,0.1, 1],
    'penalty': ['l1','l2'],
    'solver': ['saga']
     }

    # GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Optimize for accuracy
    n_jobs=-1,  # Use all available CPU cores
        )
    # Fit the grid search to the training data
    grid_search.fit(X_train, Y_train)
    # # Best estimator and its hyperparameters
    best_model = grid_search.best_estimator_

    # Calculate test accuracy
    Y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_pred) * 100
    print("Logistic Regression Test Accuracy:", test_accuracy)

    # Calculate training accuracy
    Y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(Y_train, Y_train_pred) * 100
    print("Logistic Regression Training Accuracy:", train_accuracy)
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_test_pred))
    # model_filename = "logistic_regression.pkl"
    # joblib.dump(best_model, model_filename)
    return best_model, train_accuracy , test_accuracy


# X_train = preprocessing(X_train)
# X_test = preprocessing(X_test)
# X_train,X_test = vectorizer(X_train ,X_test)
# X_train,X_test = scaling(X_train, X_test)
# X_train , X_test = maxabs_scaler(X_train , X_test)
# X_train,X_test =reduce_dimentions(X_train , X_test)
# Y_train , Y_test = label_encode(Y_train , Y_test)
# best_model_svm,train_accuracy_svm, test_accuracy_svm = classify_svm(X_train, Y_train, X_test, Y_test)
# best_model_logistic,train_accuracy_logistic, test_accuracy_logistic = classify_logistic(X_train, Y_train, X_test, Y_test)
