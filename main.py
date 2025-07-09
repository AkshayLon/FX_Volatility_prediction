import pandas as pd
import numpy as np
from data_engine import DataSet
import tensorflow as tf
import gpflow

class ResultsObject:

    def __init__(self):
        self.x = []
        self.predicted_mean = []
        self.predicted_std = []
        self.loss_profile = []
        self.test_data = []

    def update_predictions(self, f_mean, f_var):
        self.predicted_mean.append(f_mean.numpy().flatten())
        self.predicted_std.append(np.sqrt(f_var.numpy().flatten()))

    def update_loss_profile(self, test_set, f_mean):
        self.test_data.append(test_set)
        mean_values = f_mean.numpy().flatten()
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
    model = None
    optim = gpflow.optimizers.Scipy()
    X, Y, X_new = None, None, None
    for i in range(5):
        X = tf.convert_to_tensor(np.arange(10*i, (10*i)+50).reshape(-1, 1), dtype=tf.float64)
        Y = tf.convert_to_tensor(data.training_data[i].reshape(-1, 1), dtype=tf.float64)
        X_new = tf.convert_to_tensor(np.arange((10*i)+50, (10*i)+60).reshape(-1, 1), dtype=tf.float64)
        walk_forward_results.x.append(np.arange((10*i)+50, (10*i)+60))
        model = gpflow.models.GPR(
            (X, Y),
            kernel=gpflow.kernels.Matern32()
        )
        optim.minimize(model.training_loss, model.trainable_variables)
        f_mean, f_var = model.predict_f(X_new)
        walk_forward_results.update_predictions(f_mean, f_var)
        walk_forward_results.update_loss_profile(data.test_data[i], f_mean)

    return walk_forward_results