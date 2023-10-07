import pickle

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )

    # Initialize and train the logistic regression model
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    with open("model.pkl", "wb") as file:
        pickle.dump(clf, file)


if __name__ == "__main__":
    main()
