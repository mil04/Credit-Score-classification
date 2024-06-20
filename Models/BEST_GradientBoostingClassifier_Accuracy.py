import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from basic_pipeline_functions import PipelineBasic
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import joblib

data_all = pd.read_csv('../data/data.csv')

train, test = train_test_split(data_all, test_size=0.2, random_state=42 ,stratify = data_all['DEFAULT'])

X_train = train.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_train = train['DEFAULT']

X_test = test.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_test = test['DEFAULT']

clf = GradientBoostingClassifier(learning_rate=0.31154302777899784,
                                 max_depth=4,
                                 min_samples_leaf=0.27089458178978043,
                                 min_samples_split=1.0,
                                 n_estimators=25,
                                 subsample=1.0,
                                 random_state=42)

pca = PCA(n_components=23)

pipeline = Pipeline([
    ('basic_pipeline', PipelineBasic),
    ('pca', pca),
    ('classifier', clf)
])

print("______________CROSS VALIDATION_________________________________________________________")
y_pred_cv = cross_val_predict(pipeline, X_train, y_train, cv=5)

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

accuracy_cv = (conf_matrix_cv[0][0] + conf_matrix_cv[1][1]) / (conf_matrix_cv[0][0] + conf_matrix_cv[0][1] + conf_matrix_cv[1][0] + conf_matrix_cv[1][1])
print("Accuracy (cross-validation):", accuracy_cv)

print("______________TESTING_________________________________________________________")

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

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

accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / (conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1])
print("Accuracy:", accuracy)

joblib.dump(pipeline, 'gradient_boosting_classifier.joblib')
print(" saved ")

"""
______________CROSS VALIDATION_________________________________________________________
Precision for class 0 (cross-validation): 0.736318407960199
Recall for class 0 (cross-validation): 0.9694323144104804
Precision for class 1 (cross-validation): 0.6216216216216216
Recall for class 1 (cross-validation): 0.12637362637362637
Confusion Matrix (cross-validation):
[[444  14]
 [159  23]]
Accuracy (cross-validation): 0.7296875
______________TESTING_________________________________________________________
Precision for class 0: 0.7368421052631579
Recall for class 0: 0.9739130434782609
Precision for class 1: 0.625
Recall for class 1: 0.1111111111111111
Confusion Matrix:
[[112   3]
 [ 40   5]]
Accuracy: 0.73125
 saved 
"""