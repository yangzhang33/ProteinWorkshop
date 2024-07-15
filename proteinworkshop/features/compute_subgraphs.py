"""Edge construction and featurisation utils."""
import functools
from typing import List, Literal, Optional, Tuple, Union

import graphein.protein.tensor.edges as gp
import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import Protein, ProteinBatch
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph
from torch_scatter import scatter_mean
from torch_geometric.nn import radius_graph


@typechecker
def compute_subgraphs_batch(
    batch: Union[Batch, ProteinBatch],
    params,
): # -> Tuple[torch.Tensor, torch.Tensor]
    

    k_hops = params[0]
    num_subgraphs_per_protein_perc = params[1]
    #compute subgraph for a batch
    edge_index = batch.edge_index
    #edge_index = radius_graph(batch.coords_ca, r=args.cutoff, batch=batch.batch.to(device), max_num_neighbors=max_num_neighbors)
    
    subgraphs = []
    for i in range(batch.num_graphs):
        num_nodes = batch.ptr[i+1].item()-batch.ptr[i].item()
        num_subgraphs_per_protein = int(num_nodes*num_subgraphs_per_protein_perc)

        #for each protein 
        for j in range(num_subgraphs_per_protein):
            #pick a node from the protein. in batch.ptr there is the number of nodes in each protein.
            #node_pick = torch.randint(low=batch.ptr[i].item(),high=batch.ptr[i+1].item(),size=(1,))
            node_pick = batch.ptr[i].item()+j
            nodes_in_subgraph, _, _, _ = k_hop_subgraph(node_pick, k_hops, edge_index, relabel_nodes=False)
            # subgraphs.append(nodes_in_subgraph)
            subgraphs.append(nodes_in_subgraph.tolist())
            
    #compute labels
    G_c = scatter_mean(batch.pos, batch.batch, dim=0)   
    dist = []
    for i in range(len(subgraphs)):
        center_prot = G_c[batch.batch[subgraphs[i][0]]]
        # center_sub = torch.mean(batch.pos[torch.tensor(subgraphs[i])], dim=0)
        center_sub = torch.mean(batch.pos[torch.tensor(subgraphs[i]).clone().detach()], dim=0)
        dist.append(torch.norm(center_prot-center_sub))
    
    # dist / max(dist) in a batch
    max_dist = max(dist)
    dist = [d / max_dist for d in dist]
    
    # subgraphs = torch.tensor(subgraphs)
    # subgraphs = [torch.tensor(sg) for sg in subgraphs]
    ### do padding:
    # Find the maximum length of the sublists
    subgraph_lengths = [len(subgraph) for subgraph in subgraphs]
    max_len = max(len(subgraph) for subgraph in subgraphs)

    # Pad each sublist to the maximum length with a padding value (e.g., -1)
    padded_subgraphs = [subgraph + [-1] * (max_len - len(subgraph)) for subgraph in subgraphs]

    # Convert the padded sublists to a tensor
    subgraphs = torch.tensor(padded_subgraphs)

    dist = torch.tensor(dist)
    return subgraphs,dist,subgraph_lengths




@typechecker
def compute_radius_graph(
    b: Union[Data, Batch, Protein, ProteinBatch],
    cutoff_maxnumneighbors,
):
    cutoff, max_num_neighbors = cutoff_maxnumneighbors[0], cutoff_maxnumneighbors[1]
    e_index = radius_graph(b.pos, r=cutoff, batch=b.batch, max_num_neighbors=max_num_neighbors)
    return e_index