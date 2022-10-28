import os
import sys
sys.path.insert(0, os.path.abspath("NeurIPS2022-traffic4cast"))  # noqa:E402

import torch
from pathlib import Path
from functools import partial
from typing import Optional, List

from t4c22.t4c22_config import cc_dates
from t4c22.t4c22_config import load_inputs
from t4c22.t4c22_config import day_t_filter_to_df_filter
from t4c22.t4c22_config import day_t_filter_weekdays_daytime_only
from t4c22.dataloading.road_graph_mapping import TorchRoadGraphMapping
from t4c22.dataloading.t4c22_dataset import T4c22Dataset as OfficalT4c22Dataset, T4c22Competitions


class T4c22Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: Path,
    city: str,
    edge_attributes=None,
    split: str = "train",
    cachedir: Optional[Path] = None,
    limit: int = None,
    day_t_filter=day_t_filter_weekdays_daytime_only,
    competition: T4c22Competitions = T4c22Competitions.CORE,
    temporal_features: bool = False
  ):
    """Dataset for t4c22 core competition (congestion classes) for one
    city.

    Get 92 items a day (last item of the day then has x loop counter
    data at 91, 92, 93, 94 and y congestion classes at 95) I.e.
    overlapping sampling, but discarding samples going over midnight.

    Missing values in input or labels are represented as nans, use `torch.nan_to_num`.
    CC labels are shift left by one in tensor as model outputs only green,yellow,red but not unclassified and allows for direct use in `torch.nn.CrossEntropy`
      # 0 = green
      # 1 = yellow
      # 2 = red


    Parameters
    ----------
    root: basedir for data
    city: "london" / "madrid" / "melbourne"
    edge_attributes: any numerical edge attribute from `road_graph_edges.parquet`
      - parsed_maxspeed
      - speed_kph
      - importance
      - oneway
      - lanes
      - tunnel
      - length_meters
    split: "train" / "test" / ...
    cachedir: location for single item .pt files (created on first access if cachedir is given)
    limit: limit the dataset to at most limit items (for debugging)
    day_t_filter: filter taking day and t as input for filtering the data. Ignored for split=="test".
    temporal_features: add temporal features to the input
    """
    super().__init__()
    self.root: Path = root
    self.cachedir = cachedir
    self.split = split
    self.city = city
    self.limit = limit
    self.day_t_filter = day_t_filter if split != "test" else None
    self.competition = competition
    self.temporal_features = temporal_features
    self.temporal_features_suffix = "_t" if temporal_features else ""

    self.torch_road_graph_mapping = TorchRoadGraphMapping(
      city=city,
      edge_attributes=edge_attributes,
      root=root,
      df_filter=partial(day_t_filter_to_df_filter, filter=day_t_filter) if self.day_t_filter is not None else None,
      skip_supersegments=self.competition == T4c22Competitions.CORE,
    )

    # `day_t: List[Tuple[Y-m-d-str,int_0_96]]`
    # TODO most days have even 96 (rolling window over midnight), but probably not necessary because of filtering we do.
    if split == "test":
      num_tests = load_inputs(basedir=self.root, split="test", city=city, day="test", df_filter=None)["test_idx"].max() + 1
      self.day_t = [("test", t) for t in range(num_tests)]
    else:
      self.day_t = [(day, t) for day in cc_dates(self.root, city=city, split=self.split) 
                    for t in range(4, 96) if self.day_t_filter(day, t)]

  def __len__(self) -> int:
    if self.limit is not None:
      return min(self.limit, len(self.day_t))
    return len(self.day_t)
      
  def __getitem__(self, idx: int) -> torch.Tensor:  # noqa:C901
    if idx > self.__len__():
      raise IndexError("Index out of bounds")

    day, t = self.day_t[idx]

    city = self.city
    basedir = self.root
    split = self.split

    # x: 4 time steps of loop counters on nodes at t=0',+15',+30',+45'
    x = None
    if self.cachedir is not None:
      cache_file = self.cachedir / f"inputs_{self.city}_{day}_{t}{self.temporal_features_suffix}.pt"
      if cache_file.exists():
        x = torch.load(cache_file)
    if x is None:
      x = self.torch_road_graph_mapping.load_inputs_day_t(basedir=basedir, city=city, split=self.split, day=day, t=t, idx=idx)

      if self.temporal_features:
        m_day = idx // 92 if split == 'train' else 0
        day_t = torch.tensor([[m_day, t]]).type_as(x).repeat(x.size(0), 1)
        x = torch.cat([x, day_t], dim=1)
        
      if self.cachedir is not None:
        self.cachedir.mkdir(exist_ok=True, parents=True)
        torch.save(x, cache_file)
    if self.split == "test":
      return x, None

    # y: congestion classes on edges at +60'
    y = None
    if self.cachedir is not None:
      cache_file = self.cachedir / (
        f"cc_labels_{self.city}_{day}_{t}.pt" if self.competition == T4c22Competitions.CORE else f"eta_labels_{self.city}_{day}_{t}.pt"
      )
      if cache_file.exists():
        y = torch.load(cache_file)
    if y is None:
      if self.competition == T4c22Competitions.CORE:
        y = self.torch_road_graph_mapping.load_cc_labels_day_t(basedir=basedir, city=city, split=split, day=day, t=t, idx=idx)
      else:
        y = self.torch_road_graph_mapping.load_eta_labels_day_t(basedir=basedir, city=city, split=split, day=day, t=t, idx=idx)

      if self.cachedir is not None:
        self.cachedir.mkdir(exist_ok=True, parents=True)
        torch.save(y, cache_file)

    # x.size(): (num_nodes, 4) - loop counter data, a lot of NaNs!
    # y.size(): (num_edges, 1) - congestion classification data, contains NaNs.
    # edge_attr: (num_edges, len(edge_attributes)) - edge attributes, optionally
    if self.torch_road_graph_mapping.edge_attributes is None:
      return x, y

    else:
      return x, y, self.torch_road_graph_mapping.edge_attr


class T4c22TestDataset(OfficalT4c22Dataset):
  def __init__(
    self,
    root: Path,
    city: str,
    edge_attributes: Optional[List[str]] = None,
    cachedir: Optional[Path] = None,
    limit: int = None,
    temporal_features: bool = False
  ):
    """Test Dataset for t4c22 core competition (congestion classes) for one city.

    Parameters
    ----------
    root: basedir for data
    city: "london" / "madrid" / "melbourne"
    edge_attributes: any numerical edge attribute from `road_graph_edges.parquet`
            - parsed_maxspeed
            - speed_kph
            - importance
            - oneway
            - lanes
            - tunnel
            - length_meters
    cachedir: location for single item .pt files (created on first access if cachedir is given)
    limit: limit the dataset to at most limit items (for debugging)
    temporal_features: add temporal features to the input
    """
    self.split = "test"
    self.temporal_features = temporal_features
    self.temporal_features_suffix = "_t" if temporal_features else ""
    super().__init__(
      root=root,
      city=city,
      edge_attributes=edge_attributes,
      split=self.split,
      cachedir=cachedir,
      limit=limit,
      day_t_filter=None
    )

  def __getitem__(self, idx: int) -> torch.Tensor:
    if idx > self.__len__():
      raise IndexError("Index out of bounds")

    day, t = self.split, idx

    x = None
    if self.cachedir is not None:
      cache_file = self.cachedir / f"inputs_{self.city}_{day}_{t}{self.temporal_features_suffix}.pt"
      if cache_file.exists():
        x = torch.load(cache_file)
    if x is None:
      x = self.torch_road_graph_mapping.load_inputs_day_t(
        basedir=self.root, city=self.city, 
        split=self.split, day=day, 
        t=t, idx=idx)
      
      if self.temporal_features:
        day_t = torch.tensor([[day, t]]).type_as(x).repeat(x.size(0), 1)
        x = torch.cat([x, day_t], dim=1)

      if self.cachedir is not None:
        self.cachedir.mkdir(exist_ok=True, parents=True)
        torch.save(x, cache_file)

    # x.size(): (num_nodes, 4 + temporal_features)
    return x, None



if __name__ == "__main__":
  from src.utils.load import load_basedir, none_filter
  
  BASEDIR = load_basedir("/Users/mori/Codes/Data/IARAI/Traffic4cast2022/T4C_INPUTS_2022")
  CACHEDIR = Path("cache")
  city = "london"
  
  # dataset = T4c22TestDataset(root=BASEDIR, city=city,
  #                            cachedir=CACHEDIR / "processed", 
  #                            temporal_features=False)
  dataset = T4c22Dataset(root=BASEDIR, city=city,
                         cachedir=CACHEDIR / "processed", 
                         split='train',
                         day_t_filter=none_filter,
                         temporal_features=True)
  print(dataset.day_t)
  print(f"length: {len(dataset)}")
  print(f"first item: {dataset[0]}")
  