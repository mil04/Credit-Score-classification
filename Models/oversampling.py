import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import joblib

# class which deals with outliers
class OutliersReplacerBasic(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for column in X.columns:
            if column == 'SAVINGS':
                X.loc[X[column] > 2500000, column] = 2500000
            elif column == 'DEBT':
                X.loc[X[column] > 4000000, column] = 4000000
            elif column == 'T_CLOTHING_12':
                X.loc[X[column] > 32000, column] = 32000
            elif column == 'T_CLOTHING_6':
                X.loc[X[column] > 25000, column] = 25000
            elif column == 'T_HEALTH_12':
                X.loc[X[column] > 25000, column] = 25000
            elif column == 'T_HEALTH_6':
                X.loc[X[column] > 18000, column] = 18000
            elif column == 'T_TRAVEL_12':
                X.loc[X[column] > 150000, column] = 150000
            elif column == 'T_TRAVEL_6':
                X.loc[X[column] > 110000, column] = 110000
        return X


# class which drops columns which have too much random data or because of correlation > 95%
class DropColumnsBasic(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ColumnsToDrop = ['T_EXPENDITURE_12', 'T_EDUCATION_6', 'T_ENTERTAINMENT_6', 'T_GAMBLING_6',
                         'T_GROCERIES_6', 'R_UTILITIES_DEBT', 'T_HOUSING_6', 'T_TAX_6', 'T_UTILITIES_6',
                         'R_FINES', 'R_GAMBLING', 'R_EDUCATION', 'R_HOUSING']
        for column in ColumnsToDrop:
            if column in X.columns:
                X.drop(column, axis=1, inplace=True)
        return X


PipelineBasicOversample = ImbPipeline([
    ('outliers_replacer', OutliersReplacerBasic()),
    ('columns_dropper', DropColumnsBasic())
])

data_all = pd.read_csv('../data/data.csv')

data_all['CAT_GAMBLING'] = data_all['CAT_GAMBLING'].map({'No': 1, 'Low': 2, 'High': 3})

data_all = data_all.drop(['CREDIT_SCORE', 'CUST_ID'], axis=1)

X = data_all.drop('DEFAULT', axis=1)
y = data_all['DEFAULT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

log_clf = LogisticRegression(random_state=42, max_iter=300, solver='liblinear', penalty='l2', C=0.1)
selector = RFE(estimator=log_clf, step=1, n_features_to_select=13)

LR_pipeline = Pipeline([
    ('feature_selection', selector),
    ('classifier', log_clf)
])

print("______________CROSS VALIDATION_________________________________________________________")
y_pred_cv = cross_val_predict(LR_pipeline, X_train_resampled, y_train_resampled, cv=5)

precision_0_cv = precision_score(y_train_resampled, y_pred_cv, pos_label=0)
recall_0_cv = recall_score(y_train_resampled, y_pred_cv, pos_label=0)
precision_1_cv = precision_score(y_train_resampled, y_pred_cv, pos_label=1)
recall_1_cv = recall_score(y_train_resampled, y_pred_cv, pos_label=1)

print("Precision for class 0 (cross-validation):", precision_0_cv)
print("Recall for class 0 (cross-validation):", recall_0_cv)
print("Precision for class 1 (cross-validation):", precision_1_cv)
print("Recall for class 1 (cross-validation):", recall_1_cv)

conf_matrix_cv = confusion_matrix(y_train_resampled, y_pred_cv)

print("Confusion Matrix (cross-validation):")
print(conf_matrix_cv)

print("Accuracy (cross-validation):", (conf_matrix_cv[0][0] + conf_matrix_cv[1][1]) / (conf_matrix_cv[0][0] + conf_matrix_cv[0][1] + conf_matrix_cv[1][0] + conf_matrix_cv[1][1]))

print("______________TESTING_________________________________________________________")

LR_pipeline.fit(X_train_resampled, y_train_resampled)
y_pred = LR_pipeline.predict(X_test)

#accuracy matrix

precision_0 = precision_score(y_test, y_pred, pos_label=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0)
precision_1 = precision_score(y_test, y_pred, pos_label=1)
recall_1 = recall_score(y_test, y_pred, pos_label=1)

print("Precision for class 0:", precision_0)
print("Recall for class 0:", recall_0)
print("Precision for class 1:", precision_1)
print("Recall for class 1:", recall_1)

conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("Accuracy:", (conf_matrix[0][0] + conf_matrix[1][1]) / (conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]))

"""
______________CROSS VALIDATION_________________________________________________________
Precision for class 0 (cross-validation): 0.6494845360824743
Recall for class 0 (cross-validation): 0.5502183406113537
Precision for class 1 (cross-validation): 0.6098484848484849
Recall for class 1 (cross-validation): 0.7030567685589519
Confusion Matrix (cross-validation):
[[252 206]
 [136 322]]
Accuracy (cross-validation): 0.6266375545851528
______________TESTING_________________________________________________________
Precision for class 0: 0.8295454545454546
Recall for class 0: 0.6347826086956522
Precision for class 1: 0.4166666666666667
Recall for class 1: 0.6666666666666666
Confusion Matrix:
[[73 42]
 [15 30]]
Accuracy: 0.64375
"""
