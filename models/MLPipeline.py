from collections import Counter
from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, \
    average_precision_score, precision_recall_curve
from sklearn.model_selection import learning_curve
from pylab import rcParams
import seaborn as sns
from inspect import signature

class MLPipeline(object):
    """
    Class for performing feature selection based on a specific model.

    Outputs the following:
        1. Plots confusion matrix
        2. Plot learning curves
        3. Model evaluation metrics
        4. Plots feature importance
        5. Plot precision recall curves
    """

    def __init__(self, X_train,X_test, y_train, y_test, model):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model

        self.feature_importance = None
        self.y_predict = None
        # Dictionary to hold removal operations
        self.ops = {}

        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = model.predict(self.X_test)
        self.y_pred_proba = model.predict_proba(self.X_test)[:,1]
        self.y_pred_train = model.predict(self.X_train)
        self.y_pred_train_proba = model.predict_proba(self.X_train)[:,1]

    def plot_confusion_matrix(self):
        """Plots the confusion matrix"""
        rcParams['figure.figsize'] = 5.85, 4
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred)
        class_names = [0, 1]
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        # create heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("bottom")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

    def plot_learning_curves(self):
        """Plots learning curves for model validation"""
        rcParams['figure.figsize'] = 5.85, 4
        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            self.X_train,
            self.y_train,
            # Number of folds in cross-validation
            cv=5,
            # Evaluation metric
            scoring='accuracy',
            # Use all computer cores
            n_jobs=-1,
            shuffle=True,
            # 5 different sizes of the training set
            train_sizes=np.linspace(0.01, 1.0, 5))

        # Create means and standard deviations of training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Draw lines
        plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
        plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

        # Create plot
        plt.title("Learning Curves")
        plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
        plt.tight_layout()

        plt.show()

    def score(self):
        """some evaluation metrics"""
        score = {'Accuracy (Test Data)': round(self.model.score(self.X_test, self.y_test), 2),
                 'Accuracy (Train Data)': round(self.model.score(self.X_train, self.y_train), 2),
                 'F1-Score (Test Data)': round(f1_score(self.y_test, self.y_pred), 2),
                 'F1-Score (Train Data)': round(f1_score(self.y_train, self.y_pred_train), 2),
                 'Precision (Test Data)': round(precision_score(self.y_test, self.y_pred), 2),
                 'Precision (Train Data)': round(precision_score(self.y_train, self.y_pred_train), 2),
                 'Recall (Test Data)': round(recall_score(self.y_test, self.y_pred), 2),
                 'Recall (Train Data)': round(recall_score(self.y_train, self.y_pred_train), 2),
                 'ROC-AUC (Test Data)': round(roc_auc_score(self.y_test, self.y_pred_proba), 2),
                 'ROC-AUC (Train Data)': round(roc_auc_score(self.y_train, self.y_pred_train_proba), 2),
                 }

        return score

    def feature_importance_plot(self):
        """plots feature importance"""
        feat_importances = pd.Series(self.model.feature_importances_,
                                     index=self.X_train.columns)
        feat_importances.nlargest(20).plot(kind='barh')
        plt.show()

    def precision_recall_curve(self):
        """plots precision recall curve with average precision"""

        recall, precision, threshold = precision_recall_curve(self.y_test, self.y_pred_proba)
        average_precision = average_precision_score(self.y_test, self.y_pred_proba)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision));
