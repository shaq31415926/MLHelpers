# -*- coding: utf-8 -*-
import os
from collections import Counter
from itertools import chain
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import learning_curve
from sklearn.externals import joblib
# from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MLPipeline(object):
    """
    Class for performing feature selection based on a specific model.

    Outputs the following:
        1. Splits data into train and test
        2. Plots feature importance
        3. Identify low scoring/redundant features to drop
        4. Plot model scores
        5. Plot learning curves
    """

    def __init__(self, data, target, target_grouped, model, model_name):
        self.data = data
        self.target = target
        self.model = model
        self.model_name = model_name
        self.model_name2 = self.model_name.replace(" ", "_")

        # store dataframe with ids and labels - useful for prediction eval
        # self.data_copy = self.data.drop(['url_id', target_grouped, 'avg_ranking_score'], axis=1)
        # self.data_id_w_labels = self.data[['url_id', target_grouped, 'avg_ranking_score', self.target]]
        self.data_copy = self.data.drop(['url', target_grouped], axis=1)
        self.data_id_w_labels = self.data[['url', target_grouped, self.target]]

        self.target_data = self.data_copy[target]
        self.features = self.data_copy.drop([self.target], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                self.features, self.target_data,
                                                test_size=0.3, random_state=0)
        # adding scaled data as well
#        min_max_scaler = MinMaxScaler()
#        min_max_scaler.fit(self.X_train)
#        scaler = StandardScaler()
#        scaler.fit(self.X_train)
#
#        self.X_train_minmax = min_max_scaler.transform(self.X_train)
#        self.X_test_minmax = min_max_scaler.transform(self.X_test)
#
#        self.X_train_scaled = scaler.transform(self.X_train)
#        self.X_test_scaled = scaler.transform(self.X_test)

        self.feature_importance = None
        self.y_predict = None
        # self.to_drop = None
        self.datstr = time.strftime("%Y_%m_%d")
        self.datstr2 = time.strftime("%Y-%m-%d")
        # Dictionary to hold removal operations
        self.ops = {}

        # create filepath if necessary
        mypath = '../reports/figures/{}/{}'.format(self.datstr, self.model_name)
        if not os.path.isdir(mypath):
            os.makedirs(mypath)
    
        # create filepath if necessary
        mypath = '../models/{}/{}'.format(self.datstr, self.model_name)
        if not os.path.isdir(mypath):
            os.makedirs(mypath)
   
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = model.predict(self.X_test)
        self.y_pred_train = model.predict(self.X_train)

    def identify_redundant_features(self, score_threshold):
        """
        Identify the features with low importance
        """
        feature_names = list(self.features.columns)
        importances = self.model.feature_importances_

        feature_importance = pd.concat([pd.DataFrame(feature_names, columns=["features"]), pd.DataFrame(importances, columns=["score"])], axis=1)
        feature_importance = feature_importance.sort_values(by=['score'], ascending=False)

        to_drop = list(feature_importance[feature_importance['score'] < score_threshold]['features'])

        self.feature_importance = feature_importance
        self.ops['redundant_features'] = to_drop

        print("\nfeatures to drop:\n", list(to_drop))

    def plot_feature_importance(self):
        """Feature importance plot"""
        feature_names = self.features.columns
        importances = self.model.feature_importances_
        indices = np.argsort(importances)

        # plt.figure(figsize=(6, 4))
        rcParams['figure.figsize'] = 6.85, 5.7
        plt.title('Feature Importance Plot')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), feature_names[indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()

        plt.savefig('../reports/figures/{}/{}/feature_importance_plot.png'.format(self.datstr, self.model_name), bbox_inches='tight')
        plt.show()

    def plot_learning_curves(self, cv):
        """Plots learning curves for model validation"""
        rcParams['figure.figsize'] = 5.85, 4
        train_sizes, train_scores, test_scores = learning_curve(
                                                        self.model,
                                                        self.X_train,
                                                        self.y_train,
                                                        # Number of folds in cross-validation
                                                        cv=cv,
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
        plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
        plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

        # Create plot
        plt.title("Learning Curves")
        plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
        plt.tight_layout()

        plt.savefig('../reports/figures/{}/{}/learning_curves.png'.format(self.datstr, self.model_name), bbox_inches='tight')
        plt.show()

    def plot_pred_and_actual_target(self, reversefactor):
        """"
        Plots the distribution of the predicted target data against the actual
        target
        """
        rcParams['figure.figsize'] = 6.85, 4
        y_test = np.vectorize(reversefactor.get)(self.y_test)
        y_pred = np.vectorize(reversefactor.get)(self.y_pred)

        y_test = pd.DataFrame(y_test).rename(columns={0: 'actual_target'})
        y_pred = pd.DataFrame(y_pred).rename(columns={0: 'predicted_target'})

        target_combined = pd.concat([y_test, y_pred], axis=1)
        target_summary = target_combined.apply(pd.value_counts)

        target_summary = target_summary.reindex(index=[
                                                       'very_good',
                                                       'good',
                                                       'poor',
                                                       'very_poor'])

        target_summary.plot.bar(rot=0, title="Actual vs Predicted Targets")
        plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

        plt.savefig('../reports/figures/{}/{}/pred_and_target_plot.png'.format(self.datstr, self.model_name), bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, reversefactor):
        """Plots the confusion matrix"""
        rcParams['figure.figsize'] = 5.85, 5
        y_test = np.vectorize(reversefactor.get)(self.y_test)
        y_pred = np.vectorize(reversefactor.get)(self.y_pred)
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=np.unique(y_test), yticklabels=np.unique(y_test),
               title='Confusion Matrix',
               ylabel='Actual target',
               xlabel='Predicted target')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        plt.savefig('../reports/figures/{}/{}/confusion_matrix.png'.format(
                            self.datstr, self.model_name), bbox_inches='tight')

    def score_accuracy(self):
        """Accuracy score on test set"""
        # self.y_predict = model.predict(self.X_test)
        print("\naccuracy on test set: ", round(accuracy_score(self.y_test, self.y_pred), 3))

    def score_detailed(self, X, y, reversefactor, dat):
        """Detailed evaluation of model"""
        y_pred = self.model.predict(X)
        y_new = np.vectorize(reversefactor.get)(y)
        y_pred = np.vectorize(reversefactor.get)(y_pred)

        print('\n--------'+dat+' Data-----------')
        print("\n------Target Variable Distribution-------")
        print('y_'+dat.lower()+':', Counter(y_new))
        print("y_pred: ", Counter(y_pred))
        print("\n------Confusion Matrix-----")
        print(pd.crosstab(y_new, y_pred, rownames=['Actual'], colnames=['Predicted']))
        print("\naccuracy: ", round(accuracy_score(y_new, y_pred), 3))
        print("\n------Classification Report-----")
        print(classification_report(y_new, y_pred))

    def model_eval(self, X_test, X_train, selection_params):
        """
        Run all the steps to create a model report
        """
        print('\n--------Model Report-----------')
        # Plot the variable importance plot
        self.plot_feature_importance()
        # Plot accuracy on test set
        self.score_accuracy()
        # identify redundant features
        self.identify_redundant_features(selection_params['score_threshold'])

        # Find the number of features identified to drop
        self.all_identified = set(list(chain(*list(self.ops.values()))))
        self.n_identified = len(self.all_identified)
        self.to_drop = self.ops['redundant_features']

        print('%d total features out of %d identified as redundant.\n' % (self.n_identified, X_train.shape[1]))
        # score the test data
        self.score_detailed(X_test, self.y_test, selection_params['reverse_factor'], 'Test')
        # score the training data
        self.score_detailed(X_train, self.y_train, selection_params['reverse_factor'], 'Train')

    def plots_upload_s3(self, client, bucket_name, learning_curves,
                        feature_importance, selection_params):
        """
        Run all the steps to upload plots into s3
        """
        if learning_curves is True:
            # Plot learning curves
            self.plot_learning_curves(selection_params['cv'])
            client.upload_file(
                    '../reports/figures/{}/{}/learning_curves.png'.format(
                            self.datstr, self.model_name), bucket_name,
                    '{}/{}/images/learning_curves.png'.format(
                            self.datstr, self.model_name2))

        if feature_importance is True:
            # Plot feature importance plot
            self.plot_feature_importance()
            client.upload_file(
                '../reports/figures/{}/{}/feature_importance_plot.png'.format(
                        self.datstr, self.model_name), bucket_name,
                '{}/{}/images/feature_importance_plot.png'.format(
                        self.datstr, self.model_name2))

        # Plot confusion matrix
        self.plot_confusion_matrix(selection_params['reverse_factor'])
        client.upload_file(
                '../reports/figures/{}/{}/confusion_matrix.png'.format(
                        self.datstr, self.model_name), bucket_name,
                '{}/{}/images/confusion_matrix.png'.format(
                        self.datstr, self.model_name2))

        # Plot pred vs actual target
        self.plot_pred_and_actual_target(selection_params['reverse_factor'])
        client.upload_file(
                '../reports/figures/{}/{}/pred_and_target_plot.png'.format(
                        self.datstr, self.model_name), bucket_name,
                '{}/{}/images/pred_and_target_plot.png'.format(
                        self.datstr, self.model_name2))

    def evaluation_metrics(self, score, additional_info, description, image_url):
        """store evaluation metrics of all models"""
        # store evaluation metrics in store dataframe
        score.append({
         'Date': self.datstr2,
         'Index': 'Quality',
         'Test Instance': '',
         'Model Name': self.model_name,
         'Training Data Size': self.X_train.shape[0],
         'Test Data Size': self.X_test.shape[0],
         'Number of Features': self.X_test.shape[1],
         'Accuracy (Test Data)': round(self.model.score(self.X_test, self.y_test), 2),
         'Accuracy (Train Data)': round(self.model.score(self.X_train, self.y_train), 2),
         'F1-Score (Test Data)': round(f1_score(self.y_test, self.y_pred, average='weighted'), 2),
         'F1-Score (Train Data)': round(f1_score(self.y_train, self.y_pred_train, average='weighted'), 2),
         'Description': description,
         'Additional Information': additional_info,
         'Image_URL': image_url
         })

        return score

    def model_upload_s3(self, client, bucket_name):
        """upload trained model in s3"""
        # store model as pickle in defined location
        joblib.dump(self.model, '../models/{}/{}/model.pkl'.format(
                self.datstr, self.model_name))
        # upload file to s3 - need to find a shorter way of doing this
        client.upload_file(
                '../models/{}/{}/model.pkl'.format(
                        self.datstr, self.model_name), bucket_name,
                '{}/{}/model/trained_model.pkl'.format(
                        self.datstr, self.model_name2))
