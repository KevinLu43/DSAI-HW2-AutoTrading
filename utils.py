from re import L
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

# df = pd.read_csv("training.csv", header=None)
# df.columns = ["open","highest","lowest","close"]
# df["trend"] = str(0)
# df["target"] = df["open"][1487]

# for i in range(len(df["open"])-1):
#     threshold = 0.01
#     t1 = df["open"][i+1]
#     t = df["open"][i]
#     a = (t1 - t) / t
    
#     if a > threshold:
#         df["trend"][i] = str(1)
#     elif a < (-threshold):
#         df["trend"][i] = str(-1)
#     else:
#         df["trend"][i] = str(0) 
#     df["target"][i] = df["open"][i+1]

# #print(pd.value_counts(df["trend"]))
# #print(df["target"])

# df_y = df["target"]
# df_trend = df["trend"]
# df_x = df[["open", "highest", "lowest", "close"]]


# df1 = pd.read_csv("testing.csv", header=None)
# df1.columns = ["open","highest","lowest","close"]
# df1["target"] = df1["open"][19]

# for i in range(len(df1["open"])-1):
#     df1["target"][i] = df1["open"][i+1]

# #print(df_x)
# ### Regression
# model_rf = RandomForestRegressor()
# model_rf.fit(df_x, df_y)
# pred_rf = model_rf.predict(df_x)
# pred_rf_test = model_rf.predict(df1[["open","highest","lowest","close"]])

# model_lgb = LGBMRegressor()
# model_lgb.fit(df_x, df_y)
# pred_lgb = model_lgb.predict(df_x)
# pred_lgb_test = model_lgb.predict(df1[["open","highest","lowest","close"]])
# """
# ### Classify
# model_rf_c = RandomForestClassifier()
# model_rf_c.fit(df_x, df_trend)
# pred_rf_c = model_rf_c.predict(df_x)

# print(pred_rf_c, df["trend"])
# ###
# """
# """

# """

# pred_trend = []
# action = []
# now = 0
# for i in range(len(pred_lgb_test)-1):
#     threshold = 0.0001
#     t1 = pred_lgb_test[i+1]
#     t = pred_lgb_test[i]
#     a = (t1 - t) / t
    
#     if a > threshold:
#         pred_trend.append(1)
#     elif a < (-threshold):
#         pred_trend.append(-1)
#     else:
#         pred_trend.append(0)

# for i in range(len(pred_lgb_test)-1):       
#     if now == 0:
#         if pred_trend[i] == 1:
#             action.append(1)
#         elif pred_trend[i] == 0:
#             action.append(0)
#         else:
#             action.append(-1)
#     elif now == 1:
#         if pred_trend[i] == 1:
#             action.append(0)
#         elif pred_trend[i] == 0:
#             action.append(0)
#         else:
#             action.append(-1)
#     elif now == -1:
#         if pred_trend[i] == 1:
#             action.append(1)
#         elif pred_trend[i] == 0:
#             action.append(0)
#         else:
#             action.append(0)
#     now = now + action[i]
#     print(now)

# print(pred_trend)
# print(action)
