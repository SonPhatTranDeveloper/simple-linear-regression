"""
Import and pre-process the dataset
Author: Son Phat Tran
"""
import os.path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from var import PATH


# Import red wine qualification dataset
wine = pd.read_csv(os.path.join(PATH, "dataraw", "winequality-red.csv"), sep=";")

# Get the features and outputs
X = wine.iloc[:, :-1]
y = wine.iloc[:, -1]

# Scale the output
y = (y - 5) / 10

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Scale the input
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


if __name__ == "__main__":
    print(X_train)