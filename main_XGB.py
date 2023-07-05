import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
import preprocess_XGB
import random

TRAIN_PERCENT = 0.85
random.seed(101)
if __name__ == '__main__':
    data, targets = preprocess_XGB.load_dataset()

    max_key = max(data.keys())
    train_keys = random.sample(range(max_key + 1), k=int(TRAIN_PERCENT * (max_key + 1)))
    test_keys = list(set(range(max_key + 1)) - set(train_keys))

    X_train = [data[key] for key in train_keys]
    y_train = [targets[key] for key in train_keys]

    X_test = [data[key] for key in test_keys]
    y_test = [targets[key] for key in test_keys]

    model = xgb.XGBClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"F1 Score: {f1}")
    print(f"Accuracy Score: {acc}")


