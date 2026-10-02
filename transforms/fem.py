import torch
import torch_geometric as pyg
from torch_cluster import knn_graph
from torch_scatter import scatter


def compute_jacobian(pos, tets, field):
    edge_index = (
        tets_to_edge_index(tets)
        if isinstance(tets, torch.Tensor)
        else knn_graph(pos, k=13)
    )

    field_diff = field[edge_index[1]] - field[edge_index[0]]
    pos_diff = pos[edge_index[1]] - pos[edge_index[0]]

    pos_distance = torch.clamp(torch.norm(pos_diff, dim=-1, keepdim=True), min=1e-16)

    jacobian = scatter(
        (field_diff / pos_distance)[:, :, None] * (pos_diff / pos_distance)[:, None, :],
        edge_index[0].long(),
        dim=0,
        reduce="mean",
    )

    return jacobian


def tets_to_edge_index(tets):
    return pyg.utils.to_undirected(
        torch.cat((tets[0:2], tets[1:3], tets[2:4], tets[0:4:3]), dim=-1)
    )


def compute_laplacian(pos, tets, jacobian):
    edge_index = (
        tets_to_edge_index(tets)
        if isinstance(tets, torch.Tensor)
        else knn_graph(pos, k=13)
    )

    jacobian_diff = jacobian[edge_index[1]] - jacobian[edge_index[0]]
    pos_diff = pos[edge_index[1]] - pos[edge_index[0]]

    pos_distance = torch.clamp(torch.norm(pos_diff, dim=-1, keepdim=True), min=1e-16)

    laplacian = torch.sum(
        scatter(
            (jacobian_diff / pos_distance[:, :, None])
            * (pos_diff / pos_distance)[:, None, :],
            edge_index[0].long(),
            dim=0,
            reduce="mean",
        ),
        dim=-1,
    )

    return laplacian
