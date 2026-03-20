import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from model import games_up_to_2018_season_filter, season2018_filter, data_for_model, set_up_data



'''Read in model data.'''
data_df = pd.read_pickle('model_data/gamelog_5_exp_clust.pkl')
train_df, test_df = data_for_model(data_df, odds=False)
X_train, y_train, X_test, y_test = set_up_data(train_df, test_df)
data = (X_train, y_train, X_test, y_test)


gb_model = GradientBoostingClassifier(
    learning_rate=0.1, loss='exponential', max_depth=2,
    max_features=None, min_samples_leaf=2, min_samples_split=2,
    n_estimators=100, subsample=0.5)

gb_model.fit(X_train, y_train)

feat_imports = gb_model.feature_importances_

features = train_df.columns.tolist()[1:]


ft_imp_dict = {k: v for k, v in list(zip(features, feat_imports))}

ft_imp_df = pd.DataFrame.from_dict(ft_imp_dict, orient='index', columns=['Feature_Importances'])

ft_imp_df.to_csv('feature_importances.csv')
