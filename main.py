# Supress all warnings and logging messages from Tensorflow packages
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import necessary libraries
import pandas as pd
import numpy as np
from data_engine import DataSet
import tensorflow as tf
import gpflow

class RevisedMeanFunction(gpflow.functions.MeanFunction):

    def __init__(self, X, Y):
        super().__init__()
        self.mapping = dict(zip(X, Y))

    def __call__(self, x):
        x_values = x.numpy().flatten()
        y = np.vectorize(self.mapping.get)(x_values)
        return tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float64)
    
class ResultsObject:

    def __init__(self):
        self.x = []
        self.predicted_mean = []
        self.predicted_std = []
        self.loss_profile = []
        self.test_data = []

    def update_predictions(self, f_mean, f_var):
        self.predicted_mean.append(f_mean)
        self.predicted_std.append(np.sqrt(f_var))

    def update_loss_profile(self, test_set, f_mean):
        self.test_data.append(test_set)
        mean_values = f_mean
        self.loss_profile.append(np.mean(np.abs(test_set - mean_values)))

    def get_metrics(self):
        mean = np.concatenate(self.predicted_mean)
        std = np.concatenate(self.predicted_std)
        lower_band = mean-std
        upper_band = mean+std
        return {
            'x': np.concatenate(self.x),
            'mean': mean,
            'lower': lower_band,
            'upper': upper_band,
            'loss_profile': self.loss_profile
        }

def run_walkforward_analysis(data):
    walk_forward_results = ResultsObject()
    current_model = gpflow.models.GPR(
        data=(tf.convert_to_tensor(np.zeros((1, 1)), dtype=tf.float64),
              tf.convert_to_tensor(np.zeros((1, 1)), dtype=tf.float64)), 
        kernel=gpflow.kernels.Matern32(),
        mean_function=gpflow.mean_functions.Constant(0.0)
    )
    current_X, current_Y = None, None
    prediction_window = None
    optim = gpflow.optimizers.Scipy()
    for i in range(5):
        # Prepare training data for current iteration
        current_X = tf.convert_to_tensor(np.arange(10*i, (10*i)+50).reshape(-1, 1), dtype=tf.float64)
        current_Y = tf.convert_to_tensor(data.training_data[i].reshape(-1, 1), dtype=tf.float64)
        walk_forward_results.x.append(np.arange((10*i)+50, (10*i)+60))
        current_model.data = (current_X, current_Y)

        # Train model and get predictions
        prediction_window = tf.convert_to_tensor(np.arange((10*i)+50, (10*i)+110).reshape(-1, 1), dtype=tf.float64)
        optim.minimize(current_model.training_loss, current_model.trainable_variables)
        f_mean, f_var = current_model.predict_f(prediction_window)

        # Update model performance
        walk_forward_results.update_predictions(f_mean.numpy().flatten()[:10], f_var.numpy().flatten()[:10])
        walk_forward_results.update_loss_profile(data.test_data[i], f_mean.numpy().flatten()[:10])

        # change mean function for next iteration
        new_mean_function = RevisedMeanFunction(prediction_window.numpy().flatten(), f_mean.numpy().flatten())
        current_model.mean_function = new_mean_function

    return walk_forward_results

if __name__ == "__main__":
    data = DataSet('EURUSD30.csv')
    results = run_walkforward_analysis(data)
    metrics = results.get_metrics()
    print(metrics['mean'])