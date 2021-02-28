import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # for normalization
import statsmodels.api as sm

# diagnosis_map is numerical values of labels: 1 if magniglant(M) and -1 benign(B)
diagnosis_map = {'M': 1.0, 'B': -1.0}

# >> FEATURE SELECTION << #


def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped


def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped


def read_data(filename):
    data = pd.read_csv(filename)
    # drop last column (extra column added by pd)
    # and unnecessary first column (id)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
    # print(data)

    data['diagnosis'] = data['diagnosis'].map(diagnosis_map)
    # print(data['diagnosis'])

    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]
    # normalization
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)
    remove_correlated_features(X)
    remove_less_significant_features(X, Y)
    return X, Y


# read_data('data.csv')
