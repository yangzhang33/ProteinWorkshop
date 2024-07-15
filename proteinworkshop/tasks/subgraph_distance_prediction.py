from typing import Set, Union

import torch
import torch_geometric.transforms as T
from graphein.protein.tensor.data import Protein
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_scatter import scatter_mean
from torch_geometric.nn import radius_graph

class ComputeSubgraphsTransform(T.BaseTransform):
    """
    Self-supervision task 
    """

    def __init__(self, k_hops=2, max_num_neighbors=32, num_subgraphs_per_protein_perc=0.1, cutoff=10):
        """Initialise the transform.


        """
        self.k_hops = k_hops
        self.max_num_neighbors = max_num_neighbors
        self.num_subgraphs_per_protein_perc = num_subgraphs_per_protein_perc
        self.cut_off = cutoff

    @property
    def required_batch_attributes(self) -> Set[str]:
        """
        Returns the set of attributes that this transform requires to be
        present on the batch object for correct operation.


        """
        return {"pos", "batch", "ptr", "edge_index"}

    def __call__(
        self, x: Union[Data, Protein]
    ) -> Union[Data, Protein]:
        # print(x)
        subgraphs = []
        num_nodes = len(x.x)
        # print(num_nodes)
        num_subgraphs_per_protein = int(num_nodes * self.num_subgraphs_per_protein_perc)
        batch = torch.zeros(num_nodes, dtype=torch.int64)
        # # For each protein
        pos = x.coords[:, 1, :]
        edge_index = radius_graph(pos, r=self.cut_off, batch=batch, max_num_neighbors=self.max_num_neighbors)
        # print(edge_index)
        # print(num_nodes, edge_index.shape)
        for j in range(num_subgraphs_per_protein):
            node_pick = j
            # print(j, edge_index)
            # start = time.time()
            nodes_in_subgraph, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_pick, self.k_hops, edge_index, relabel_nodes=False)
            # print(nodes_in_subgraph.shape)
            subgraphs.append(nodes_in_subgraph)
        # print(len(subgraphs))
        # # Compute labels
        # G_c = torch.mean(pos, dim=0)
        # dist = []
        # for i in range(len(subgraphs)):
        #     center_sub = torch.mean(x.coords[i, 1, :], dim=0)
        #     dist.append(torch.norm(G_c - center_sub))
        # dist = torch.tensor(dist)
        x.subgraphs = [1]
        x.subgraph_distances = [2]
        return x
    



