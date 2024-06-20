from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

#class which deals with outliers
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

# class which drops columns which have to much random data or because of correlation > 95%
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

# class which deals with categorical features
class TransformCategoricalFeaturesBasic(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        if 'CUST_ID' in X.columns:
            X.drop("CUST_ID", axis = 1, inplace = True)
        X['CAT_GAMBLING'] = X['CAT_GAMBLING'].map({'No': 1, 'Low': 2, 'High': 3})
        return X
    

PipelineBasic = Pipeline([
    ('categorical_features_transformer', TransformCategoricalFeaturesBasic()),
    ('outliers_replacer', OutliersReplacerBasic()),
    ('columns_dropper', DropColumnsBasic())
])
