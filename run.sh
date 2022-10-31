# data imputation
ipython src/imputation/gp_impute.py -- --city london
ipython src/imputation/gp_impute.py -- --city madrid
ipython src/imputation/gp_impute.py -- --city melbourne

# data preparation
ipython src/preparation/prepare_train_test_arrays.py
ipython src/preparation/extract_missing_index.py
ipython src/preparation/missing_data_split.py
ipython src/preparation/speed_processing.py

# feature extraction
ipython src/feature_extraction/static_features.py -- --city london
ipython src/feature_extraction/static_features.py -- --city melbourne
ipython src/feature_extraction/static_features.py -- --city madrid
ipython src/feature_extraction/loop_features_fully_missing.py -- --city london --construct_test
ipython src/feature_extraction/loop_features_fully_missing.py -- --city melbourne --construct_test
ipython src/feature_extraction/loop_features_fully_missing.py -- --city madrid --construct_test
ipython src/feature_extraction/loop_features_fully_missing.py -- --city london --calculate_missing
ipython src/feature_extraction/speed_features_fully.py -- --city london --create_x_sg --knn_p 1
ipython src/feature_extraction/speed_features_fully.py -- --city melbourne --create_x_sg --knn_p 1
ipython src/feature_extraction/speed_features_fully.py -- --city madrid --create_x_sg --knn_p 1
ipython src/feature_extraction/speed_features_missing.py
ipython src/feature_extraction/knn_features_fully.py -- --city london --knn_p 1
ipython src/feature_extraction/knn_features_fully.py -- --city melbourne --knn_p 1
ipython src/feature_extraction/knn_features_fully.py -- --city madrid --knn_p 1
ipython src/feature_extraction/knn_features_fully.py -- --city london --knn_p 2
ipython src/feature_extraction/knn_features_fully.py -- --city melbourne --knn_p 2
ipython src/feature_extraction/knn_features_fully.py -- --city madrid --knn_p 2
ipython src/feature_extraction/knn_features_missing.py

# model fitting and prediction
ipython src/models/create_missing_df.py
ipython src/models/lgb_missing.py
ipython src/models/lgb_combine_p.py

# make submission
ipython src/submission/make_submission.py -- --submission_name custom_model_name
