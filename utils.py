import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch import cuda
from torch import device
from torch import from_numpy
from torch import nn 
from torch.optim import Adam 
from torch.utils.data import TensorDataset, dataloader


if cuda.is_available():
    device = device("cuda")
else:
    device = device("cpu")

class Trader():
    
    def __init__(self):
        
        return
#print(df)
    def plot_figure(self):
        plt.figure(figsize=(10,6))
        plt.plot(df["open"],label="Open", marker="*")
        plt.plot(df["highest"],label="Highest")
        plt.plot(df["lowest"],label="Lowest")
        plt.plot(df["close"],label="Close", marker=".")
        plt.title("Stock Price")
        plt.legend()
        plt.show()

    def train(self, df):

        return

    def predict(self, df):
        
        return
    
if __name__ == '__main__':
    df = pd.read_csv("testing.csv", header=None)
    df.columns = ["open","highest","lowest","close"]
    df["trend"] = 0

    print(df)
    Trader.plot_figure(df)