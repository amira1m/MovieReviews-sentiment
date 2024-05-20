from main import  *




# # vectorizing processed data
def load_vectorizer(X):
    tf_idf_path = "tf_idf_vectorizer.pkl"
    tf_idf_model = joblib.load(tf_idf_path)
    X = tf_idf_model.transform(X)
    return X



def load_standard(X):
    standard_scaler_path = "standard_scaler.pkl"
    standard_scaler_model = joblib.load(standard_scaler_path)
    X = standard_scaler_model.transform(X)
    return X

def load_maxabs(X):
    maxabs_scaler_path = "maxabs_scaler.pkl"
    maxabs_scaler_model = joblib.load(maxabs_scaler_path)
    X = maxabs_scaler_model.transform(X)
    return X

def load_reducer(X):
    reducer_path = "dimentions_reducer.pkl"
    reducer_model = joblib.load(reducer_path)
    X = reducer_model.transform(X)
    return X


def load_encoder(Y):
    label_encoder_path = "label_encoder.pkl"
    label_encoder_model = joblib.load(label_encoder_path)
    Y = label_encoder_model.transform(Y)
    return Y

def load_svm(X , Y):
    svm_path = "svm.pkl"
    svm_model = joblib.load(svm_path)
    Y_pred = svm_model.predict(X)
    Y_pred = load_encoder(Y_pred)
    accuracy_svm = accuracy_score(Y, Y_pred) * 100
    print("svm model accuracy:", accuracy_svm)
def load_logistic(X , Y):
    logistic_path = "logistic_regression.pkl"
    logistic_model = joblib.load(logistic_path)
    Y_pred_logistic = logistic_model.predict(X)
    Y_pred_logistic=load_encoder(Y_pred_logistic)
    accuracy_logistic = accuracy_score(Y, Y_pred_logistic) * 100
    print("logistic model accuracy:", accuracy_logistic)

X = preprocessing(X)
X = load_vectorizer(X)
X= load_standard(X)
X = load_maxabs(X)
X= load_reducer(X)
Y = load_encoder(Y)
load_svm(X ,Y)
load_logistic(X , Y)
