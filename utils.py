
from numpy.lib.shape_base import _replace_zero_by_x_arrays
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

class Trader:
    def __init__(self):
        self.model = LGBMRegressor()
        self.now = 0
    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        prediction = self.model.predict(x)
        return prediction

    def predict_action(self, x, prediction, threshold):
        pred_trend = 0
        action = 0
        threshold = threshold
        t1 = prediction
        t = x
        a = (t1 - t) / t
            
        if a > threshold:
            pred_trend = 1
        elif a < (-threshold):
            pred_trend = -1
        else:
            pred_trend = 0   
        if self.now == 0:
            if pred_trend == 1:
                action = 1
            elif pred_trend == 0:
                action = 0
            else:
                action = -1
        elif self.now == 1:
            if pred_trend == 1:
                action = 0
            elif pred_trend == 0:
                action = 0
            else:
                action = -1
        elif self.now == -1:
            if pred_trend == 1:
                action = 1
            elif pred_trend == 0:
                action = 0
            else:
                action = 0
        self.now = self.now + action
        return str(action)
