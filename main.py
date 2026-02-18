import csv
import pandas as pd

# Passenger ID has no predictive capacity but will be used as an index, so it will be dropped from the features,
# For Name, titles are all that is needed (Mr, Mrs, Miss, Dr, etc...), 
# Age has null values so it will be filled with median age, 
# cabin has null values (only about 70% have a value) and it is not clear how to fill them, so it will be dropped,
# Ticket has no clear pattern and it is not clear how to extract useful information from it, so it will be dropped,
def preProcess(data):
    data["Title"] = data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    data.drop("Name", axis=1, inplace=True)
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data.drop("Cabin", axis=1, inplace=True)
    data.drop("Ticket", axis=1, inplace=True)


def main():
    data = pd.read_csv("train.csv")
    preProcess(data)
    print(data)

if __name__ == "__main__":
    main()