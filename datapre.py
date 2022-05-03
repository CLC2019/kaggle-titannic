import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders.ordinal import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import titanic_config as config

train = pd.read_csv("data/train/train.csv")
test = pd.read_csv("data/test/test.csv")
#print(train.info)
train[["Deck", "Cabin_Num", "Side"]] = train.Cabin.str.split("/", expand=True)
test[["Deck", "Cabin_Num", "Side"]] = test.Cabin.str.split("/", expand=True)

train[["Group", "Group_Num"]] = train.PassengerId.str.split("_", expand=True)
test[["Group", "Group_Num"]] = test.PassengerId.str.split("_", expand=True)

#replacing NaN values
for i in train.columns:
    if train[i].isna().sum() > 0:
        print(f"{i}: {train[i].isna().sum()}")

print("\n", "test NaNs", "\n")
for i in test.columns:
    if test[i].isna().sum() > 0:
        print(f"{i}: {test[i].isna().sum()}")


for i in train.columns:
    print(f"{i}: {train[i].nunique()}")

# Make a list of columns that only have a couple of values to replace with mode
mode_list = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]

# Replace these columns NaN values with the mode.
for i in mode_list:
    train[i] = train[i].fillna(train[i].mode()[0])
    test[i] = test[i].fillna(train[i].mode()[0])  # Fill in the test with same values

for i in train.columns:
    if train[i].isna().sum() > 0:
        print(f"{i}: {train[i].isna().sum()}")

# Make a list of numeric columns to replace with the median
median_list = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

for i in median_list:
    train[i] = train[i].fillna(train[i].median())
    test[i] = test[i].fillna(train[i].median())  # Fill in the test with same values

for i in train.columns:
    if train[i].isna().sum() > 0:
        print(f"{i}: {train[i].isna().sum()}")

# These last columns will have NaN values replaced with a value that indicates that there wasn't a value.
train["Cabin"] = train["Cabin"].fillna(f"{train.Deck}/-1/{train.Side}")
train["Name"] = train["Name"].fillna("No name listed")
train["Cabin_Num"] = train["Cabin_Num"].fillna("-1")

test["Cabin"] = test["Cabin"].fillna(f"{train.Deck}/-1/{train.Side}")
test["Name"] = test["Name"].fillna("No name listed")
test["Cabin_Num"] = test["Cabin_Num"].fillna("-1")

for i in train.columns:
    if train[i].isna().sum() > 0:
        print(f"{i}: {train[i].isna().sum()}")

oe = OrdinalEncoder()
scaler = StandardScaler()

transform_pipe = Pipeline([("Encoder", oe), ("Scaler", scaler)])

X = train.drop("Transported", axis=1)
y = train["Transported"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

transform_pipe.fit(X_train)
X_train_transform = transform_pipe.transform(X_train)
X_test_transform = transform_pipe.transform(X_test)
print(X_train_transform.shape, X_test_transform.shape)


# Change the train and test data to type tensor.
class trainData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


train_data = trainData(torch.tensor(X_train_transform, dtype=torch.float, requires_grad=True),
                       torch.tensor(y_train.to_numpy(), dtype=torch.int64, requires_grad=False))


class testData(Dataset):
    def __init__(self, x_data):
        self.x_data = x_data

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return len(self.x_data)


test_data = trainData(torch.tensor(X_test_transform, dtype=torch.float),
                      torch.tensor(y_test.to_numpy(), dtype=torch.int64, requires_grad=False))

def get_loader():
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1)
    return train_loader, test_loader