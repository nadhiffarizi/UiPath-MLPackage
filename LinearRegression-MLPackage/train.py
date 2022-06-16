import pandas as pd
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import joblib


class Main(object):
    def __init__(self):
        self.model_path = './linear_model.pickle'
        self.model = joblib.load(self.model_path)

    def train(self, training_directory):
        X, y = self.load_data(os.path.join(training_directory, 'train.csv'))
        # retrain the model
        self.model.fit(X, y)

    def evaluate(self, evaluate_directory):
        x_test, y_test = self.load_data(os.path.join(evaluate_directory, 'evaluate.csv'))
        # return accuracy score
        acc = self.model.score(x_test, y_test)
        return acc

    def save(self):
        joblib.dump(self.model, self.model_path)

    def load_data(self, path):
        df = pd.read_csv(path)
        X = df[['feature1']]
        y = df[['feature2']]
        return X, y