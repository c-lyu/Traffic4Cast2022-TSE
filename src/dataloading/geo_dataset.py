from typing import Optional

import torch
from torch import Tensor
import torch_geometric
import numpy as np

from t4c22.dataloading.t4c22_dataset_geometric import T4c22GeometricDataset
from t4c22.t4c22_config import load_road_graph


class T4c22GeomExtDataset(torch_geometric.data.Dataset):
  """Dataset for t4c22 core competition (congestion classes) for one
  city.

  Get 92 items a day (last item of the day then has x loop counter
  data at 91, 92, 93, 94 and y congestion classes at 95) I.e.
  overlapping sampling, but discarding samples going over midnight.

  Missing values in input or labels are represented as nans, use `torch.nan_to_num`.

  CC labels are shift left by one in tensor as model outputs only green,yellow,red 
  but not unclassified and allows for direct use in `torch.nn.CrossEntropy`
      # 0 = green
      # 1 = yellow
      # 2 = red

  Parameters
  ----------
  dataset: official T4c22GeometricDataset
  x_imputed: imputed array of flow
  new_edge_attr: new edge attributes to be appended
  """
  def __init__(
    self,
    dataset: T4c22GeometricDataset,
    x_imputed: Optional[np.ndarray] = None,
    new_edge_attr: Optional[Tensor] = None,
  ):
    super().__init__()

    self.dataset = dataset
    self.num_day_t = len(dataset)
    self.day_t = dataset.day_t
    self.root = dataset.root
    self.cachedir = dataset.cachedir
    self.split = dataset.split
    self.city = dataset.city
    self.day_list = sorted(set([day for day, t in self.day_t]))
    
    _, df_nodes, _ = load_road_graph(self.root, self.city)
    self.nodes_valid = torch.tensor(df_nodes.counter_info.ne("").values, dtype=torch.bool)
    
    self.edge_index = dataset[0].edge_index
    self.edge_attr = dataset[0].edge_attr
    self.x_raw = dataset[0].x
    
    self.x_imputed = x_imputed
    self.edge_attr_new = self.update_edge_attr(self.edge_attr, new_edge_attr)
    
  def update_edge_attr(self, edge_attr, new_edge_attr):
    if new_edge_attr is not None:
      ea = torch.cat([edge_attr, new_edge_attr], dim=1)
    else:
      ea = edge_attr
    return ea
    
  def len(self) -> int:
    return len(self.dataset)
    
  def get(self, idx: int) -> torch_geometric.data.Data:
    if self.split == 'test':
      data_raw = self.dataset[idx]
      x = data_raw.x
      data = torch_geometric.data.Data(
        x=x, 
        edge_index=self.edge_index, 
        y=None, 
        edge_attr=self.edge_attr_new)
    else:
      y = self.dataset[idx].y
      if self.x_imputed is not None:
        day, t = self.day_t[idx]
        x = torch.full_like(self.x_raw, torch.nan).type_as(self.x_raw)
        day_idx = self.day_list.index(day)
        day_t_idx = day_idx * 95 + t
        x[self.nodes_valid] = torch.tensor(
          self.x_imputed[:, day_t_idx - 4 : day_t_idx]).type_as(x)
      else:
        x = self.dataset[idx].x
      data = torch_geometric.data.Data(
        x=x, 
        edge_index=self.edge_index,
        y=y,
        edge_attr=self.edge_attr_new)
    return data
