from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

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

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if 'CUST_ID' in X.columns:
            X.drop("CUST_ID", axis=1, inplace=True)
        X['CAT_GAMBLING'] = X['CAT_GAMBLING'].map({'No': 1, 'Low': 2, 'High': 3})
        return X

# class which deals with numerical features
class TransformNumerical(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Столбцы для Box-Cox трансформации
        box_cox_columns = ['INCOME', 'SAVINGS', 'DEBT', 'R_SAVINGS_INCOME',
                           'R_DEBT_INCOME', 'R_DEBT_SAVINGS', 'T_CLOTHING_12', 'T_CLOTHING_6',
                           'R_CLOTHING_INCOME', 'R_CLOTHING_SAVINGS',
                           'R_CLOTHING_DEBT', 'T_EDUCATION_12', 'T_EDUCATION_6',
                           'R_EDUCATION_INCOME', 'R_EDUCATION_SAVINGS', 'R_EDUCATION_DEBT',
                           'R_ENTERTAINMENT_SAVINGS',
                           'R_ENTERTAINMENT_DEBT', 'T_FINES_12', 'T_FINES_6',
                           'R_FINES_INCOME', 'R_FINES_SAVINGS', 'R_FINES_DEBT', 'T_GAMBLING_12',
                           'T_GAMBLING_6', 'R_GAMBLING_INCOME', 'R_GAMBLING_SAVINGS',
                           'R_GAMBLING_DEBT', 'T_GROCERIES_12', 'T_GROCERIES_6',
                           'R_GROCERIES_SAVINGS', 'R_GROCERIES_DEBT',
                           'T_HEALTH_12', 'T_HEALTH_6',
                           'R_HEALTH_DEBT', 'T_HOUSING_12', 'T_HOUSING_6',
                           'R_HOUSING_SAVINGS', 'R_HOUSING_DEBT',
                           'T_TAX_12', 'T_TAX_6', 'R_TAX_INCOME', 'R_TAX_SAVINGS',
                           'R_TAX_DEBT', 'T_TRAVEL_12', 'T_TRAVEL_6',
                           'R_TRAVEL_SAVINGS', 'R_TRAVEL_DEBT',
                           'T_UTILITIES_12', 'T_UTILITIES_6',
                           'R_UTILITIES_DEBT',
                           'R_EXPENDITURE_INCOME']

        yeo_johnson_columns = ['T_ENTERTAINMENT_12', 'T_ENTERTAINMENT_6', 'R_ENTERTAINMENT_INCOME',
                               'R_GROCERIES_INCOME', 'R_HEALTH_INCOME', 'R_HEALTH_SAVINGS',
                               'R_UTILITIES_SAVINGS', 'T_EXPENDITURE_12', 'T_EXPENDITURE_6',
                               'R_EXPENDITURE_SAVINGS', 'R_EXPENDITURE_DEBT']

        for column in box_cox_columns:
            X[column] = PowerTransformer(method = 'box-cox').fit_transform(X[column].values.reshape(-1, 1)+0.00000001)

        for column in yeo_johnson_columns:
            X[column] = PowerTransformer(method='yeo-johnson').fit_transform(X[column].values.reshape(-1, 1)+0.00000001)

        return X

PipelineBasic = Pipeline([
    ('categorical_features_transformer', TransformCategoricalFeaturesBasic()),
    ('outliers_replacer', OutliersReplacerBasic()),
    ('column_transformations', TransformNumerical()),
    ('columns_dropper', DropColumnsBasic())
])
