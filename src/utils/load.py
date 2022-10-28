import json
import numpy as np
import pandas as pd
from pathlib import Path
from numpy import ndarray
from datetime import date
from typing import Union, Tuple
from functools import partial
import importlib.resources as pkg_resources

import t4c22
from t4c22.t4c22_config import load_road_graph
from t4c22.t4c22_config import day_t_filter, DAY_T_FILTER


def load_path_config(config: Union[Path, str]="config.json"):
    """Load basic paths from config file.
    
    Parameters
    ----------
    config: path to config file; defaults to `Path("config.json")`
    
    Returns
    -------
    config: dict
    """
    config = Path(config) if isinstance(config, str) else config
    config = json.load(open(config))
    for k, v in config.items():
        config[k] = Path(v)
    return config


cfg = load_path_config()


def load_basedir(fn: Union[Path, str] = None, pkg=t4c22) -> Path:
    """Load t4c22 data basedir from central config file.

    Parameters
    ----------
    fn: json file with BASEDIR entry; defaults to `Path("t4c22_config.json")`
    pkg: if to load from resource path; defaults to `t4c22`

    Returns
    -------
    """
    if isinstance(fn, str):
        config = {"BASEDIR": fn}
    elif fn is None:
        fn = Path("t4c22_config.json")
    else:
        if pkg is None:
            with fn.open() as f:
                config = json.load(f)
        else:
            with pkg_resources.path(pkg, fn) as p:
                with open(p) as f:
                    config = json.load(f)
    return Path(config["BASEDIR"])


def get_edge_node_idx(basedir: Union[Path, str], city: str) -> Tuple[ndarray, ndarray]:
    """Retrieve indices of u/v of edges.

    Parameters
    ----------
    basedir: path to data directory
    city: name of city

    Returns
    -------
    edge_u_idx, edge_v_idx: indices of u/v of edges
    """
    df_edges, df_nodes, _ = load_road_graph(basedir, city)
    node_idx = {node_id: idx for idx, node_id in enumerate(df_nodes["node_id"])}

    edge_u_idx = df_edges["u"].map(node_idx).values
    edge_v_idx = df_edges["v"].map(node_idx).values
    return edge_u_idx, edge_v_idx


class SkipDayFilter(DAY_T_FILTER):
    """A Day T Filter for interwoven selection of dates.

    Parameters
    ----------
    start: start date
    end: end date
    n_daysskip: number of days to skip between consecutive days
    n_dayscont: number of days to select consecutively
    """

    def __init__(self, start, end, n_daysskip=7, n_dayscont=7):
        super().__init__()
        self.day_whitelist = self._get_day_whitelist(start, end, n_daysskip, n_dayscont)

    def _get_day_whitelist(self, start, end, n_daysskip=7, n_dayscont=7):
        whitelist = pd.date_range(start=start, end=end, freq="D").astype(str).to_list()
        n_daysperiod = n_daysskip + n_dayscont
        whitelist = [
            v for i, v in enumerate(whitelist) if i % n_daysperiod < n_dayscont
        ]
        return whitelist

    def __call__(self, day, t):
        f = partial(day_t_filter, day_whitelist=self.day_whitelist)
        return f(day, t)


class SkipDayDaytimeFilter(DAY_T_FILTER):
    """A Day T Filter for interweaving selection of dates at daytime.

    Parameters
    ----------
    start: start date
    end: end date
    n_daysskip: number of days to skip between consecutive days
    n_dayscont: number of days to select consecutively
    init_n_daysskip: number of days to skip at the beginning
    """

    def __init__(self, start, end, n_daysskip=7, n_dayscont=7, init_n_daysskip=0):
        super().__init__()
        self.day_whitelist = self._get_day_whitelist(
            start, end, n_daysskip, n_dayscont, init_n_daysskip
        )
        self.t_whitelist = set(range(6 * 4, 22 * 4))

    def _get_day_whitelist(
        self, start, end, n_daysskip=7, n_dayscont=7, init_n_daysskip=0
    ):
        whitelist = pd.date_range(start=start, end=end, freq="D").astype(str).to_list()
        n_daysperiod = n_daysskip + n_dayscont
        whitelist = [
            v
            for i, v in enumerate(whitelist[init_n_daysskip:])
            if i % n_daysperiod < n_dayscont
        ]
        return whitelist

    def __call__(self, day, t):
        day_in = day in self.day_whitelist
        t_in = t in self.t_whitelist
        return day_in and t_in


"""A Day T Filter for all dates."""
none_filter: DAY_T_FILTER = day_t_filter


def get_first_date(city, fmt="dt"):
    city_first_date_str = {
        "london": "2019-07-01",
        "madrid": "2021-06-01",
        "melbourne": "2020-06-01",
    }
    city_first_date_dt = {
        "london": date(2019, 7, 1),
        "madrid": date(2021, 6, 1),
        "melbourne": date(2020, 6, 1),
    }
    if fmt == "str":
        return city_first_date_str[city]
    elif fmt == "dt":
        return city_first_date_dt[city]


def make_submission_df(city, base_dir):
    edges, _, _ = load_road_graph(base_dir, city)
    edges = edges[["u", "v"]]
    submission = pd.DataFrame(
        {
            "logit_green": np.zeros(len(edges)),
            "logit_yellow": np.zeros(len(edges)),
            "logit_red": np.zeros(len(edges)),
            "u": edges.u.values,
            "v": edges.v.values,
        }
    )

    test = pd.read_parquet(base_dir / f"test/{city}/input/counters_test.parquet")
    len_test = test.test_idx.nunique()

    submission = pd.concat([submission] * len_test)
    submission["test_idx"] = np.repeat(np.arange(len_test), len(edges))
    return submission[
        ["u", "v", "test_idx", "logit_green", "logit_yellow", "logit_red"]
    ]
