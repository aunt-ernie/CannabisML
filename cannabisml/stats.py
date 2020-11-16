"""Features normalization and ANOVA feature selection"""

import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.api as sm
from patsy.builtins import Q
from statsmodels.stats.multitest import multipletests

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, SelectPercentile


def normalize(df_features, df_covar):
    """Normalize data using a GLM.

    Parameters
    ----------
    df_features : pd.DataFrame
        a dataframe with anatomical or diffusion measures
    df_covar : pd.DataFrame
        a dataframe with variables to adjust data
        'Age': continuous
        'Sex': 'Male' or 'Female'
        'Site': 'MRN' or 'MEG'
        'ICV': continuous

    Returns
    -------
    df_adj_feats : pd.DataFrame
        predicted GLM values of shape (n_subject, n_features),
        adjusted for age, sex, site, and ICV
    df_uni_stats : pd.DataFrame
        booleans of shape (n_features, n_covariates) specifiying
        which feature/covariate univariate test is significant
        following FDR-correction.
    feature_names : list
        list of all feature names
    """
    feature_names = list(df_features.columns)
    df = pd.concat([df_covar, df_features], axis=1)
    df_uni_stats = pd.DataFrame()
    df_residuals = pd.DataFrame()

    # Univariate statistics
    for idx, covar in enumerate(df.loc[:, :df_covar.columns[-1]]):
        pvalues = []
        for feature in df.loc[:, df_features.columns[0]:]:
            # t-test if covariate is nominal, pearson corr test if continuous
            if covar == 'Site':
                df_meg = df[df.Site == 'MEG']
                df_mrn = df[df.Site == 'MRN']
                _, pval = stats.ttest_ind(df_meg[feature], df_mrn[feature], equal_var=False)
            elif covar == 'Sex':
                df_m = df[df.Sex == 'Male']
                df_f = df[df.Sex == 'Female']
                _, pval = stats.ttest_ind(df_m[feature], df_f[feature], equal_var=False)
            else:
                _, pval = stats.pearsonr(df[covar], df[feature])

            pvalues.append(pval)

        # FDR-corrected pvalues
        pvals_corrected = multipletests(pvalues, alpha=0.05, method='fdr_bh')[0]
        df_uni_stats[covar] = pvals_corrected

    # GLM correction using residuals
    for feature in df_features:
        formula = f'Q("{feature}") ~ Site + Sex + ICV + Age'
        model = sm.formula.glm(formula=formula, data=df, family=sm.families.Gaussian()).fit()
        adjusted_vals = model.predict()
        residual = np.subtract(df[feature].values, adjusted_vals)
        df_residuals[feature] = residual

    return df_residuals, df_uni_stats, feature_names


class ANOVASelection(BaseEstimator, TransformerMixin):
    """ Custom feature selection using F-tests allowing shape of transformed data
    to known to create_model.

    Parameters
    ----------
    BaseEstimator : sklearn.base.BaseEstimator
        Base class for all estimators in scikit-learn
    TransformerMixing : sklearn.base.TransformerMixin
        Mixin class for all transformers in scikit-learn.
    percentile : int
        take top percentile of features

    Returns
    -------
    percentile : int
        take top percentile of features
    m : sklearn.feature_selection.SelectPercentile
        Select features according to a percentile of the highest scores.
    scores_ : array-like of shape (n_features,)
        Scores of features.
    X_new : array-like of shape (n_samples, n_features)
    """
    def __init__(self, percentile=10):
        self.percentile = percentile
        self.m = None
        self.X_new = None
        self.scores_ = None

    def fit(self, X, y):
        self.m = SelectPercentile(f_classif, self.percentile)
        self.m.fit(X,y)
        self.scores_ = self.m.scores_
        return self

    def transform(self, X):
        global X_new
        self.X_new = self.m.transform(X)
        X_new = self.X_new
        return self.X_new


def top_hundred(residuals, feature_names, labels):
    """ Determines top 100 features using F-statistics.

    Parameters
    ----------
    residuals : np.array or list-like
        GLM normalized residuals
    features_names: np.array or list-like
        feature names the correspond to residuals
    labels : np.array or list-like
        class labels for each observation

    Returns
    -------
    top_features : np.array
        the top 100 features
    """
    F, _ = f_classif(residuals, labels)
    F_dict = {}
    for idx, fstat in enumerate(F):
        F_dict[feature_names[idx]] = fstat

    F_dict_sort = {k: v for k, v in sorted(F_dict.items(), key=lambda item: item[1], reverse=True)}
    top_features = np.array(list(F_dict_sort.keys()))[:100]

    return top_features


def calc_performance(tp, tn, fp, fn):
    """Calculate model overal model performance.

    Parameters
    ----------
    tp : list
        number of true positive in each fold
    tn : list
        number of true negative in each fold
    fp : list
        number of false positive in each fold
    fn : list
        number of false positives in each fold

    Returns
    -------
    """

    tp = np.sum(tp)
    tn = np.sum(tn)
    fp = np.sum(fp)
    fn = np.sum(fn)

    if fp != 0 and fn != 0:
        # This is most likely
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        NPV = tn/(tn+fn)
        PPV = tp/(tp+fp)
    elif fp != 0 and fn == 0:
        # Likely overfitting
        sensitivity = 1
        specificity = tn/(tn+fp)
        NPV = 1
        PPV = tp/(tp+fp)
    elif fp == 0 and fn != 0:
        # Likely overfitting
        sensitivity = tp/(tp+fn)
        specificity = 1
        PPV = 1
        NPV = tn/(tn+fn)
    if fp == 0 and fn == 0:
        # Perfect classification - yeah right...
        sensitivity = 1
        specificity = 1
        NPV = 1
        PPV = 1

    return [sensitivity, specificity, PPV, NPV]
