"""Weighted ordinary linear regression for the real-estate homework.

The estimator is linear in the coefficients of log(price). No tree models are used.
All learned preprocessing statistics are fitted on the training fold only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class Version3Features(TransformerMixin, BaseEstimator):
    def __init__(self, trend='blend', mixing=1.0):
        self.trend = trend
        self.mixing = mixing

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # These transformations use no statistics and no target values.
        x = X.drop(columns=['id', 'price', 'date'], errors='ignore').copy().fillna(-1)
        date = pd.to_datetime(X['date'], format='%Y-%m')
        t = (date.dt.year - 2011) * 12 + date.dt.month - 1
        numeric = pd.DataFrame({
            'time': t, 'time2': np.maximum(t - 11, 0),
            'logarea': np.log(x.area), 'area': x.area,
            'arearoom': x.area / x.rooms,
        })
        if self.trend in ['linear', 'blend']:
            numeric = numeric.drop(columns='time2')
        if self.trend == 'blend':
            numeric['time'] = (self.mixing * np.minimum(t, 18) / 18
                               + (1 - self.mixing) * np.log1p(t / 6) / np.log(4))
        for c in [f'kw{i}' for i in range(1, 14)]:
            numeric[c] = x[c]
        categorical = x[['street_id', 'build_tech', 'g_lift', 'floor',
                         'rooms', 'metro_dist', 'balcon', 'n_photos']].astype(str)
        groups = [
            ['rooms', 'area'], ['floor', 'build_tech'], ['rooms', 'build_tech'],
            ['rooms', 'area', 'build_tech'], ['metro_dist', 'build_tech'],
            ['g_lift', 'build_tech'], ['balcon', 'build_tech'],
        ]
        for columns in groups:
            categorical['_'.join(columns)] = x[columns].astype(str).agg('_'.join, axis=1)
        return pd.concat([numeric, categorical], axis=1)


class Version3LinearRegressor(RegressorMixin, BaseEstimator):
    """OLS for log-price with training weights (price / median(price)) ** power."""
    def __init__(self, weight_power=0.5, trend='blend', mixing=1.0):
        self.weight_power = weight_power
        self.trend = trend
        self.mixing = mixing

    def fit(self, X, y):
        target = np.asarray(y, dtype=float)
        if not (np.isfinite(target).all() and (target > 0).all()):
            raise ValueError('Prices must be finite and positive.')
        # Inspect dtypes only. Imputer, scaler and OHE are fitted by the pipeline.
        example = Version3Features(self.trend, self.mixing).transform(X.iloc[:1])
        num_cols = example.select_dtypes(include='number').columns.tolist()
        cat_cols = example.select_dtypes(exclude='number').columns.tolist()
        numeric = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        categorical = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ])
        preprocess = ColumnTransformer([
            ('numeric', numeric, num_cols),
            ('categorical', categorical, cat_cols),
        ], sparse_threshold=1.0)
        self.pipeline_ = Pipeline([
            ('features', Version3Features(self.trend, self.mixing)),
            ('preprocess', preprocess),
            ('regressor', LinearRegression(tol=1e-6)),
        ])
        weights = (target / np.median(target)) ** self.weight_power
        self.pipeline_.fit(X, np.log(target), regressor__sample_weight=weights)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, 'pipeline_')
        return np.exp(self.pipeline_.predict(X))
