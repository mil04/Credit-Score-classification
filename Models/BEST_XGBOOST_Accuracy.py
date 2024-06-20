import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from basic_pipeline_functions import PipelineBasic
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import joblib

data_all = pd.read_csv('../data/data.csv')

train, test = train_test_split(data_all, test_size=0.2, random_state=42, stratify = data_all['DEFAULT'])

X_train = train.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_train = train['DEFAULT']

X_test = test.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_test = test['DEFAULT']

xgb_clf = xgb.XGBClassifier(
    colsample_bytree=0.5,
    gamma=0.2783388624832722,
    learning_rate=0.08868316171946664,
    max_depth=3,
    min_child_weight=1,
    n_estimators=17,
    reg_alpha=1e-09,
    reg_lambda=6.738226501770893e-06,
    subsample=0.9330907199955734,
    random_state=42
)

XGB_pipeline = Pipeline([
    ('basic_pipeline', PipelineBasic),
    ('pca', PCA(n_components=20)),
    ('classifier', xgb_clf)
])

print("______________CROSS VALIDATION_________________________________________________________")
y_pred_cv = cross_val_predict(XGB_pipeline, X_train, y_train, cv=5)

precision_0_cv = precision_score(y_train, y_pred_cv, pos_label=0)
recall_0_cv = recall_score(y_train, y_pred_cv, pos_label=0)
precision_1_cv = precision_score(y_train, y_pred_cv, pos_label=1)
recall_1_cv = recall_score(y_train, y_pred_cv, pos_label=1)

print("Precision for class 0 (cross-validation):", precision_0_cv)
print("Recall for class 0 (cross-validation):", recall_0_cv)
print("Precision for class 1 (cross-validation):", precision_1_cv)
print("Recall for class 1 (cross-validation):", recall_1_cv)

conf_matrix_cv = confusion_matrix(y_train, y_pred_cv)

print("Confusion Matrix (cross-validation):")
print(conf_matrix_cv)

print("Accuracy (cross-validation):", (conf_matrix_cv[0][0] + conf_matrix_cv[1][1]) / (conf_matrix_cv[0][0] + conf_matrix_cv[0][1] + conf_matrix_cv[1][0] + conf_matrix_cv[1][1]))

print("______________TESTING_________________________________________________________")

XGB_pipeline.fit(X_train, y_train)
y_pred = XGB_pipeline.predict(X_test)

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

joblib.dump(XGB_pipeline, 'xgboost.joblib')
print(" saved ")
"""
/Users/katebokhan/anaconda3/envs/6.86x/bin/python "/Users/katebokhan/Desktop/mlProject1/Final version/Models/BEST_XGBOOST_Accuracy.py"
______________CROSS VALIDATION_________________________________________________________
Precision for class 0 (cross-validation): 0.7330016583747927
Recall for class 0 (cross-validation): 0.9650655021834061
Precision for class 1 (cross-validation): 0.5675675675675675
Recall for class 1 (cross-validation): 0.11538461538461539
Confusion Matrix (cross-validation):
[[442  16]
 [161  21]]
Accuracy (cross-validation): 0.7234375
______________TESTING_________________________________________________________
Precision for class 0: 0.7302631578947368
Recall for class 0: 0.9652173913043478
Precision for class 1: 0.5
Recall for class 1: 0.08888888888888889
Confusion Matrix:
[[111   4]
 [ 41   4]]
Accuracy: 0.71875
 saved 

Process finished with exit code 0

"""