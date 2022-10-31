import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.feature_extraction.feature_combination import feature_combine
from src.utils.load import cfg

PROCESSED = cfg["PROCESSED"]

city = "london"
num_t_per_day = 64
num_val = 0

nan_all = np.load(PROCESSED / city / "nan_percent_all.npy")
num_missing = nan_all[nan_all > 80].shape[0]
x_error, x_correct = pd.read_pickle(PROCESSED / city / "error_index.pckl")
x_train, y_train, x_test = pd.read_pickle(PROCESSED / city / "missing_df_p1.pckl")

# combine features
x_train = feature_combine(x_train)
x_test = feature_combine(x_test)

# reduce the space
x_train = x_train.astype("float32")
# x_val = x_val.astype("float32")
x_test = x_test.astype("float32")
y_train = y_train.astype("float32")
# y_val = y_val.astype("float32")

lgb_model = lgb.LGBMRegressor(
    num_leaves=42,
    objective="l1",
    first_metric_only=True,
    n_estimators=5000,
    learning_rate=0.1,
    max_depth=7,
    colsample_bytree=0.7,
    seed=42,
    n_jobs=-1,
)

lgb_model.fit(
    x_train,
    y_train,
    eval_set=[(x_train, y_train)],
    eval_metric=["mae", "rmse"],
    eval_names=["train"],
    categorical_feature=["sg_id"],
    callbacks=[lgb.early_stopping(20, first_metric_only=True), lgb.log_evaluation(10)],
)

joblib.dump(lgb_model, PROCESSED / "checkpoints" / f"lgb_full_missing_model_{city}.pkl")

y_hat_train = lgb_model.predict(x_train)
y_hat_train = np.reshape(y_hat_train, (num_missing - num_val, -1))

y_train = np.reshape(np.array(y_train), (num_missing - num_val, -1))

# make prediction
y_hat = lgb_model.predict(x_test)
y_hat = y_hat.reshape((6, -1))

np.savez_compressed(PROCESSED / "london_missing_y_eta_hat", y_hat)
