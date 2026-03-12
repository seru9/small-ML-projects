from Dataset import downloadDataset, prepareData, remodelData
import LinearRegression
import LogisticRegression
import numpy as np
def main():
    print("Loading the dataset...")
    path = downloadDataset()
    x_train, x_test, y_train, y_test = prepareData(path)

    print("Starting Linear Regression model....")
    y_train_bin = np.array(remodelData(y_train))
    y_test_bin = np.array(remodelData(y_test))
    model = LinearRegression.LinearRegression(lr = 0.0001)
    model.fit(x_train, y_train_bin)
    raw_predictions = model.predict(x_test) # there might be overflow of gradient (values are being poweref by 2 f.e)
    binary_predictions = np.where(raw_predictions >= 0.5, 1, 0)
    # Calculate Accuracy
    accuracy = np.mean(binary_predictions == y_test_bin)

    print(f"\nLinear Regression Accuracy: {accuracy * 100:.2f}%")
    model = LogisticRegression.LogisticRegression(learning_rate=0.001)
    model.fit(x_train, y_train_bin)
    predictions = model.predict(x_test)
    accuracy = np.mean(predictions == y_test_bin)
    print(f"\nLogistic Regression Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()