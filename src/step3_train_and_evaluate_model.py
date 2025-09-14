# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def train_and_evaluate(df, num_features, cat_features, target):
    # Split the data into features and target variable
    X = df[num_features + cat_features]
    y = df[target]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the training and validation sets
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    # Add the prediction column to the training and validation dataframes
    X_train['prediction'] = train_preds
    X_val['prediction'] = val_preds

    # Print the mean absolute error of the model on the training and validation data
    print("Training Mean Absolute Error: ", mean_absolute_error(y_train, train_preds))
    print("Validation Mean Absolute Error: ", mean_absolute_error(y_val, val_preds))

    # Return the trained model and the training and validation data
    return model, X_train, X_val, y_train, y_val


if __name__ == "__main__":
    pass
