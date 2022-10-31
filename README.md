# Traffic4Cast 2022 - TSE
Solution of team TSE to NeurIPS2022-Traffic4cast Challenge

- [Traffic4Cast 2022 - TSE](#traffic4cast-2022---tse)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Checkpoints](#checkpoints)
  - [Feature Engineering](#feature-engineering)
    - [Prerequisites](#prerequisites)
    - [Static network features](#static-network-features)
    - [Loop counts features](#loop-counts-features)
    - [Speed features](#speed-features)
    - [KNN label features](#knn-label-features)
    - [Feature combination](#feature-combination)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Installation

Necessary packages needed for running the scripts are included in `requirements.txt`.
In addition, the official t4c package have to be installed in advance.

```bash
pip install -r requirements.txt
```

## Usage

The scripts used for data imputation, data preparation, feature extraction and model prediction are included in `run.sh`.

```bash
sh run.sh
```

## Checkpoints

The model checkpoints are included in the folder `processed/checkpoints`.

| Checkpoints                       | Description                                                 |
|-----------------------------------|-------------------------------------------------------------|
| `lgb_1+nr2_model_london.pkl`        | London model with Mahattan and normed Euclidean distance    |
| `lgb_1+nr2_model_madrid.pkl`        | Madrid model with Mahattan and normed Euclidean distance    |
| `lgb_1+nr2_model_melbourne.pkl`     | Melbourne model with Mahattan and normed Euclidean distance |
| `lgb_1+p2_model_london.pkl`         | London model with Mahattan and Euclidean distance           |
| `lgb_1+p2_model_madrid.pkl`         | Madrid model with Mahattan and Euclidean distance           |
| `lgb_1+p2_model_melbourne.pkl`      | Melbourne model with Mahattan and Euclidean distance        |
| `lgb_full_missing_model_london.pkl` | London model for samples with high missing rate             |

## Feature Engineering

### Prerequisites
The codes of feature engineering are included in the folder `src/feature_extraction`.
Please note that, before running the codes within this folder to extract features, the scripts within the `src/preparation` folder should be run first to prepare all required inputs. Those scripts should be run as follows.

- `prepare_train_test_arrays.py`: restructure the raw loop dataset, the imputed loop dataset, and the y labels in the eta task.
- `extract_missing_index.py`: calculate the missing rate of loop data for each observation (time step).
- `missing_data_split.py`: construct the *support* set and *train* set for the observations with high missing rate in loop data in London specifically.
- `speed_processing.py`: processing the speed data, which will be used for extracting speed-based features.

### Static network features

See `static_features.py`.

- Number of nodes involved in the supersegment (SG)
- Length of SG
- Number of oneway edges in the SG
- Statistics of the speed limits of edges in the SG: mean, std, 25, 50, 75 percentiles, min, max
- Haversine distance between SG OD
- For SG $i$: Shortest/design travel time = $\sum_{j \in SG_i} \frac{\text{length}_j}{\text{MaxSpeed}_j}$
- Statistics of the $y$ values of all samples under consideration (all nn)
- Percentage of $(- \infty, 1800]$, $(1800, 2400]$ and $(2400, \infty)$ in the y query set for each SG

### Loop counts features

See `loop_features_fully_missing.py`.

- Sum, mean, std of loop counts (at nodes) within SG
- Number of loops with values (at each interval)

### Speed features

See `speed_features_fully.py`. Free flow speed and median speed of a SG is defined as the mean free flow speed and mean median speed of the edges involved. $k \in [1,2,5,10,50]$ below.

- Mean, std of the free flow speed, median speed of $k$ nearest neighbors
- Mean, std of the edge volume classes percentage/distribution of $k$ nearest neighbors

### KNN label features

See `knn_features_eng.py` and `knn_features_manipulate.py`. $k \in [2,5,10,30,50,100]$ below.

- Statistics of $y$ values of the $k$ nearest neighbors: mean, std, 25, 50, 75 percentiles, min, max

### Feature combination
We also combine (difference, addition, quotient) multiple aforementioned features together to construct more powerful features. This step is carried on in the model training script.


## Citation

```
@misc{tse-t4c22,
  title     = {Similarity-based Feature Extraction for Large-scale Sparse Traffic Forecasting},
  author    = {Wu, Xinhua and Lyu, Cheng and Lu, Qing-Long and Mahajan Vishal},
  year      = 2022,
  month     = {Oct},
  url       = {https://github.com/c-lyu/Traffic4Cast2022-TSE},
  language  = {en}
}
```

## Acknowledgements

This repository is based on the official repository of the competition [NeurIPS2022-traffic4cast](https://github.com/iarai/NeurIPS2022-traffic4cast).
