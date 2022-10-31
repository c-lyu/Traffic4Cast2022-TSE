import numpy as np
from pathlib import Path

from src.utils.load import cfg

PROCESSED = cfg["PROCESSED"]

nan_all = np.load(PROCESSED / "london" / "nan_percent_all.npy")
x_impute = np.load(PROCESSED / "london" / "X.npz")["arr_0"]
y_eta = np.load(PROCESSED / "london" / "y_eta.npz")["arr_0"]
y_eta = np.reshape(y_eta, (len(x_impute), -1))

x_support = x_impute[nan_all < 80]
x_train = x_impute[nan_all > 80]

y_support = y_eta[nan_all < 80]
y_train = y_eta[nan_all > 80]

np.savez_compressed(PROCESSED / "london" / "X_support_missing", x_support)
np.savez_compressed(PROCESSED / "london" / "X_train_missing", x_train)

np.savez_compressed(PROCESSED / "london" / "y_support_eta_missing", y_support)
np.savez_compressed(PROCESSED / "london" / "y_train_eta_missing", y_train)
