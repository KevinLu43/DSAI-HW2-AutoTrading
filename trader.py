from utils import Trader

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    import pandas as pd
    training_data = pd.read_csv(args.training, header=None)
    training_data.columns = ["open","highest","lowest","close"]
    training_data["target"] = training_data["open"][len(training_data)-1]

    for i in range(len(training_data["open"])-1):
        training_data["target"][i] = training_data["open"][i+1]
    train_y = training_data["target"]
    train_x = training_data[["open", "highest", "lowest", "close"]]
    trader = Trader()
    trader.train(train_x, train_y)
    
    testing_data = pd.read_csv(args.testing, header=None)
    testing_data.columns = ["open", "highest", "lowest", "close"]
    prediction = trader.predict(testing_data)
    
    threshold = 0.0005
    output = []
    for row in range(len(testing_data)-2):
        # We will perform your action as the open price in the next day.
        action = trader.predict_action(testing_data["open"][row], prediction[row], threshold)
        output.append(action)
    
    pd.DataFrame(output).to_csv(args.output, index=0)
    
            # this is your option, you can leave it empty.
            