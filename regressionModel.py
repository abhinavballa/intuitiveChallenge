import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

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
    Creates a scikit-learn pipeline for data preprocessing and regression.

    The pipeline consists of:
    1. Imputer: Uses the median to handle missing values
    2. Regressor: Linear Regression for price prediction

    Returns:
    sklearn.pipeline.Pipeline: Pipeline object with imputer, scaler, and Linear regression
    '''

    return Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler()), ('regressor', LinearRegression())])

def pipeline_2():
    '''
    Creates a scikit-learn pipeline for data preprocessing and regression.

    The pipeline consists of:
    1. Imputer: Uses the median to handle missing values
    2. Regressor: Gradient Boosting Regression

    Returns:
    sklearn.pipeline.Pipeline: Pipeline object with imputer, scaler, and Gradient Boosting regression
    '''

    return Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler()), ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))])

def lasso(X, y):
    '''
    Selects important features using Lasso regression.
    
    Args:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target variable
    
    Returns:
    list: Names of selected important features
    '''
    lasso_cv = LassoCV(cv=50, random_state=42)
    lasso_cv.fit(X, y)
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': np.abs(lasso_cv.coef_)})
    important_features = feature_importance[feature_importance['importance'] > 0]['feature'].tolist() #Select features with non-zero coefficients
    
    return important_features

def train(df):
    df = remove_outliers(data = df, variable = 'charges', upper = np.percentile(df['charges'], 95))
    df['charges'] = np.log1p(df['charges']) #to address positive skew in charges column
    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    regression_pipeline = pipeline_2()
    regression_pipeline.fit(X_train, y_train)

    train_pred = np.expm1(regression_pipeline.predict(X_train))
    val_pred = np.expm1(regression_pipeline.predict(X_val))
    train_mse = mean_squared_error(np.expm1(y_train), train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(np.expm1(y_train), train_pred)
    val_mse = mean_squared_error(np.expm1(y_val), val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(np.expm1(y_val), val_pred)

    # kf = KFold(n_splits=100, shuffle=True, random_state=42)
    # regression_pipeline = pipeline_2()
    # scores = cross_val_score(regression_pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')
    # rmse_scores = np.sqrt(-scores)
    # r2_scores = cross_val_score(regression_pipeline, X, y, cv=kf, scoring='r2')
    # return [rmse_scores.mean(), r2_scores.mean()]
    
    return [train_rmse, train_r2, val_rmse, val_r2]
    


df = pd.read_csv('support2[84].csv')
reg_exclude = ['aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'death', 'sfdm2', 'edu', 'temp', 'income'] #remove charges later
categorical_features = ['sex', 'dzgroup', 'dzclass', 'race', 'diabetes', 'dementia', 'ca', 'adlp']
df_update = df.drop(columns=reg_exclude)


df_encode, ohe_feat = encode_features(df_update, categorical_features)
#df_encode['charges'] = df_encode['charges'].fillna(df_encode['charges'].mean())
df_encode = df_encode.dropna(subset=['charges']) #Dropped rows with invalid or missing charge values. Tried to impute with the mean, but it seemed to increase error and drop R^2

X = df_encode.drop('charges', axis=1)
y = df_encode['charges']

#Lasso steps
# main_features = lasso(X, y)
# print(main_features)
# df_encode = df_encode[main_features + ['charges']]

res = train(df_encode)

print("Training RMSE:", res[0])
print("Training R2:", res[1])
print("Validation RMSE:", res[2])
print("Validation R2:", res[3])


'''
First Run:
- Because RMSE and R^2 are around the 54000 to 56000 and 0.70-0.71 respectively for the training and validation data, we can say that overfitting doesn't seem to be a problem.
- A R^2 score of 0.7 means that around 30% of the variance of actual charges(dependent variable) can be explained by the features (independent variables)
- charges column is skewed

After removing outliers(capping charge values at 95th percentile):
- brought RMSEs to around 20000 and R^2 to 77% to 83%



After implementing Lasso regression:
- Training RMSE around 13000-14000
- Validation RMSE around 16000
- Training R^2 around 90%
- Validation R^2 around 88-90%
Only selects 2 features that have an nonzero weight coefficient

After Log transformation of charges column and using Gradient Boosting instead of regular Linear Regression:
- RMSE and R^2 have improved greatly:
Example from a run:
Training RMSE: 6593.435759302411
Training R2: 0.9796609982689571
Validation RMSE: 12570.180204446493
Validation R2: 0.9339641513277029

Insights:
- I'm seeing that the model predicts better on instances with lower costs.
- There seems to be overfitting, as training error is considerably less than validation error


Next Steps:
- Implement k-folds cross-validation to prevent overfitting

'''