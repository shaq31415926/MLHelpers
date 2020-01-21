import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams


class RemoveOutliers(object):
    """Identifying and removing outliers"""

    def __init__(self, data):
        self.input_data = data
        self.numerical_cols = list(self.input_data.select_dtypes(include=['int64', 'float64']))

    def outliers(self, var):
        q1, q3 = np.percentile(self.input_data[var], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5*iqr)
        upper_bound = q3 + (1.5*iqr)

        return lower_bound, upper_bound

    def outliers_summary(self):
        """data frame to summarise outliers for a sense check"""
        results = []
        for i in self.numerical_cols:
            lower_bound, upper_bound = self.outliers(i)
            results.append({'var_name': i,
                            'max': self.input_data[i].max(),
                            'upper bound': upper_bound,
                            '# of contents above upper bound': len(self.input_data[self.input_data[i] > upper_bound]),
                            'min': self.input_data[i].min(),
                            'lower bound': lower_bound,
                            '# of contents below lower bound': len(self.input_data[self.input_data[i] < lower_bound]),
                            })
        result_df = pd.DataFrame(data=results)
        result_df = result_df[['var_name',
                               'max',
                               'upper bound',
                               '# of contents above upper bound',
                               'min',
                               'lower bound',
                               '# of contents below lower bound']].sort_values(['# of contents above upper bound'], ascending=False)
        print('Summary of Outliers')

        return result_df

    def outliers_removal(self, remove_outliers_var):
        """remove all outliers"""
        input_data_copy = self.input_data
        rcParams['figure.figsize'] = 5.85, 2.4
        for i in remove_outliers_var:
            print('\n', i)
            sns.boxplot(input_data_copy[i]);
            plt.show()
            lower_bound, upper_bound = self.outliers(i)
            input_data_copy = input_data_copy[input_data_copy[i].between(lower_bound, upper_bound, inclusive=True)]
            print('new shape of data:', input_data_copy.shape)

        return input_data_copy
