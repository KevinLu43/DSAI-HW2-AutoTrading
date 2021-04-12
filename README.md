# DSAI-HW2-AutoTrading

# Overview
The purpose is to **Maximize** the profit.

Data Collection
---
The training_data and testing_data are provided. 

Methodology
---
The idea is making the predicion of the next day open price firest. Second, decide the trend of the next day, then take the action according to the predicted trend.
Considering the state of holding stock, the avalible action is limited. If holding 1 unit already, only "hold" and "sell" could be considered and so on.
The prediction model is build by GBM.