"""
MUTAG dataset loader from HuggingFace.
Converts to PyTorch Geometric Data objects.
"""

import torch
from torch_geometric.data import Data
from datasets import load_dataset


def load_mutag():
    """
    Load MUTAG from HuggingFace and convert to PyG Data list.
    Returns: list of torch_geometric.data.Data (187 graphs)
    """
    raw = load_dataset("graphs-datasets/MUTAG")

    graphs = []
    for split in raw.values():
        for item in split:
            x = torch.tensor(item['node_feat'], dtype=torch.float)

            edge_index = torch.tensor(item['edge_index'], dtype=torch.long)
            # HuggingFace stores as list of [src, dst] pairs → transpose to [2, E]
            if edge_index.dim() == 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.t().contiguous()

            edge_attr = torch.tensor(item['edge_attr'], dtype=torch.float) \
                if item.get('edge_attr') is not None else None

            y_val = item['y']
            if isinstance(y_val, list):
                y_val = y_val[0]
            y = torch.tensor([int(y_val)], dtype=torch.long)

            graphs.append(Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                num_nodes=item.get('num_nodes', x.size(0))
            ))

    return graphs
