## <a name='fea_eng'></a>Feature Engineering

- [Feature Engineering](#fea_eng)
    - [Prerequisites](#prerequisites)
    - [Static features](#static_fea)
    - [Loop features](#loop_fea)
    - [Speed features](#speed_fea)
    - [knn features](#knn_fea)
    - [Feature conmbination](#fea_combination)

### <a name='prerequisites'></a>Prerequisites
Please note that, before running the codes within this folder to extracting features, the scripts within the `../preparation` folder should be run first to prepare all required inputs. Those scripts should be run as follows.
- `prepare_train_test_arrays.py`: restructure the raw loop dataset, the imputed loop dataset, and the y labels in the eta task.
- `extract_missing_index.py`: calculate the missing rate of loop data for each observation (time step).
- `missing_data_split.py`: construct the *support* set and *train* set for the observations with high missing rate in loop data in London specifically.
- `speed_processing.py`: processing the speed data, which will be used for extracting speed-based features.

### <a name='static_fea'></a>Static network features

See `static_features.py`.

- Number of nodes involved in the supersegment (SG)
- Length of SG
- Number of oneway edges in the SG
- Statistics of the speed limits of edges in the SG: mean, std, 25, 50, 75 percentiles, min, max
- Haversine distance between SG OD
- For SG i: Shortest/design travel time = $\sum_{j \in SG_i} \frac{length_j}{MaxSpeed_j}$
- Statistics of the $y$ values of all samples under consideration (all nn)
- Percentage of ($-\infty$, 1800], (1800, 2400] and (2400, $\infty$) in the y query set for each SG

### <a name='loop_fea'></a>Loop counts features

See `loop_features_fully_missing.py`.

- Sum, mean, std of loop counts (at nodes) within SG
- Number of loops with values (at each interval)

### <a name='speed_fea'></a>Speed features

See `speed_features_fully.py`. Free flow speed and median speed of a SG is defined as the mean free flow speed and mean median speed of the edges involved. $k \in [1,2,5,10,50]$ below.

- Mean, std of the free flow speed, median speed of $k$ nearest neighbors
- Mean, std of the edge volume classes percentage/distribution of $k$ nearest neighbors

### <a name='knn_fea'></a>F4 $k$nn label features

See `knn_features_eng.py` and `knn_features_manipulate.py`. $k \in [2,5,10,30,50,100]$ below.

- Statistics of $y$ values of the $k$ nearest neighbors: mean, std, 25, 50, 75 percentiles, min, max

### <a name='fea_combination'></a>Feature combination
We also combine (difference, addition, quotient) multiple aforementioned features together to construct more powerfule features. This step is carried on in the model training script.
