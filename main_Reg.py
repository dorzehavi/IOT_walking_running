import preprocess_Reg
import random
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

TRAIN_PERCENT = 0.85
random.seed(101)
if __name__ == '__main__':
    data, targets = preprocess_rfRegressor.load_dataset()

    max_key = max(data.keys())
    train_keys = random.sample(range(max_key + 1), k=int(TRAIN_PERCENT * (max_key + 1)))
    test_keys = list(set(range(max_key + 1)) - set(train_keys))

    X_train = [data[key] for key in train_keys]
    y_train = [targets[key] for key in train_keys]

    X_test = [data[key] for key in test_keys]
    y_test = [targets[key] for key in test_keys]

    model = RandomForestRegressor(n_estimators=100)

    gd_boost = GradientBoostingRegressor(n_estimators=100)
    gd_boost.fit(X_train, y_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    squared_errors = (y_pred - y_test) ** 2

    # Calculate the mean squared error
    mse = np.mean(squared_errors)

    # Calculate RMSE
    rmse = np.sqrt(mse)

    print("RMSE for RfR:", rmse)

    y_pred = gd_boost.predict(X_test)

    squared_errors = (y_pred - y_test) ** 2

    # Calculate the mean squared error
    mse = np.mean(squared_errors)

    # Calculate RMSE
    rmse = np.sqrt(mse)

    print("RMSE for GDB:", rmse)
