import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

#Define helper functions

def one_hot_encode(data, column):
    '''
    Performs one-hot encoding on specified categorical columns in a DataFrame.

    Args:
    data (pd.DataFrame): Input DataFrame
    column (list): List of column names to encode

    Returns:
    pd.DataFrame: Original DataFrame joined with new one-hot encoded columns
    list: Names of the new one-hot encoded features

    '''
    cols = data[column]
    onehot = OneHotEncoder()
    onehot.fit(cols)
    oneArr = onehot.transform(cols).todense()
    df = pd.DataFrame(data = oneArr, columns = onehot.get_feature_names_out(), index = data.index)
    return data.join(df), onehot.get_feature_names_out()

def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):

    '''
    Removes outliers from a DataFrame based on specified bounds for a variable.

    Args:
    data (pd.DataFrame): Input DataFrame
    variable (str): Column name to check for outliers
    lower (float): Lower bound (default: -inf)
    upper (float): Upper bound (default: inf)

    Returns:
    pd.DataFrame: DataFrame with outliers removed
    '''
    return data[(data[variable] > lower) & (data[variable] <= upper)]

def calculate_fpr(y_true, y_pred):
    '''
    Calculates the False Positive Rate (FPR) for classification.

    Args:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels

    Returns:
    float: False Positive Rate
    '''
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if (fp+tn) > 0:
        return fp / (fp + tn)
    else:
        return 0
   
def encode_features(df, categorical_features):
    '''
    One-hot encodes the categorical features in a DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame
    categorical_features (list): List of categorical column names

    Returns:
    pd.DataFrame: DataFrame with encoded features
    list: Names of the new encoded features
    '''
    encoded_feature_names = []    
    for col in categorical_features:
        df, new_features = one_hot_encode(df, [col])
        encoded_feature_names.extend(new_features)
        df = df.drop(columns=[col])
    return df, encoded_feature_names

def pipeline():
    '''
    Creates a scikit-learn pipeline for data preprocessing and classification.

    The pipeline consists of:
    1. Imputer: Uses the median to handle missing values
    2. Classifier: Logistic Regression for binary classification 

    Returns:
    sklearn.pipeline.Pipeline: Pipeline object with imputer, scaler, and logistic regression
    '''
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('classifier', LogisticRegression())])


def train_classifier(df):
    X = df.drop('hospdead', axis=1)
    y = df['hospdead']

    #Split data (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #Fit classification pipeline
    classification_pipeline = pipeline()
    classification_pipeline.fit(X_train, y_train)
    #Make predictions
    train_pred = classification_pipeline.predict(X_train)
    val_pred = classification_pipeline.predict(X_val)
    print(train_pred)
    
    #Calculate probabilities for ROC-AUC
    train_prob = classification_pipeline.predict_proba(X_train)[:, 1]
    val_prob = classification_pipeline.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_prob)
    val_auc = roc_auc_score(y_val, val_prob)

    train_metrics = [accuracy_score(y_train, train_pred), precision_score(y_train, train_pred), recall_score(y_train, train_pred), calculate_fpr(y_train, train_pred), train_auc]
    val_metrics = [accuracy_score(y_val, val_pred), precision_score(y_val, val_pred), recall_score(y_val, val_pred), calculate_fpr(y_val, val_pred), val_auc]

    return train_metrics, val_metrics

df = pd.read_csv('support2[84].csv')

#Exclude non-useful features
class_exclude = ['aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'death', 'sfdm2', 'charges']
categorical_features = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'diabetes', 'dementia', 'ca', 'adlp']
df_update = df.drop(columns=class_exclude)
df_encode, ohe_feat = encode_features(df_update, categorical_features)
df_encode = df_encode.dropna(subset=['hospdead'])
train_metrics, val_metrics = train_classifier(df_encode)

print("Training Accuracy:", train_metrics[0])
print("Training Precision:", train_metrics[1])
print("Training Recall:", train_metrics[2])
print("Training FPR:", train_metrics[3])
print("Training AUC:", train_metrics[-1])
print("Validation Accuracy:", val_metrics[0])
print("Validation Precision:", val_metrics[1])
print("Validation Recall:", val_metrics[2])
print("Validation FPR:", val_metrics[3])
print("Validation AUC:", val_metrics[-1])

'''
- Training and Validation metrics are similar, and in some cases validation is higher, so overfitting doesn't seem like a problem
- Precision is around 90%, so 90% of "True" predictions are correct
- Recall should be higher if model used clinically, as only 90% of actual death scenarios were predicted correctly.
- 3-5% FPR means that there are a small amount of predicted dead patients who do not have a hospital death
- 99% AUC means that the model can distinguish between positive and negative classifications excellently

'''