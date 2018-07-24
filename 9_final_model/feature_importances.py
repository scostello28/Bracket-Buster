import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
import pickle

def feature_importance(model, x_test, y_test):
   columns=x_test.columns
   original_f1 = f1_score(y_test, model.predict(x_test))
   test_f1s = []
   col_prec_dict = {}
   col_std_dict = {}
   for column_ind in range(x_test.shape[1]):
       test_f1s = []
       x_temp = x_test.values
       for num in range(100):
           indexes = np.random.permutation(np.arange(x_test.shape[0]))
           column = x_temp[:,column_ind]
           x_temp[ : ,column_ind] = column[indexes]
           test_f1s.append(f1_score(y_test, model.predict(x_temp)))
       col_prec_dict[columns[column_ind]] = original_f1-np.mean(test_f1s)
       col_std_dict[columns[column_ind]] = np.std(test_f1s)
   return col_prec_dict, col_std_dict

def plot_feature_importance(model, x_test, y_test):
   column_importance_dict, column_std_dict = feature_importance(model, x_test, y_test)

   importance_df = pd.DataFrame.from_dict(column_importance_dict, orient='index')
   importance_df['Change in F1'] = importance_df
   importance_df = importance_df.reset_index()
   importance_df['Feature'] = importance_df['index']
   importance_df = importance_df.sort_values(by=['Change in F1'])
   sns.barplot(x='Change in F1', y='Feature',data=importance_df)
   plt.tight_layout()
   plt.title('F1 Change Per Feature\nFeature Importance')
   plt.show()

   variance_df = pd.DataFrame.from_dict(column_std_dict, orient='index')
   variance_df['STD in F1'] = variance_df
   variance_df = variance_df.reset_index()
   variance_df['Feature'] = variance_df['index']
   variance_df = variance_df.sort_values(by=['STD in F1'])
   sns.barplot(x='STD in F1', y='Feature',data=variance_df)
   plt.tight_layout()
   plt.title('F1 STD Per Feature\nFeature Importance')
   plt.show()

if __name__ == '__main__':

    '''import model'''
    gb_model = 'fit_models/gb_fit_model.pkl'

    pickled_model = gb_model

    with open(pickled_model, 'rb') as f:
        model = pickle.load(f)

    '''import data'''
    data = pd.read_pickle('final_model_data/gamelog_exp_clust.pkl')
