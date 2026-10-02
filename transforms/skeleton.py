import potpourri3d as pp3d
import torch
import torch_geometric as pyg
from torch_cluster import knn
from torch_scatter import scatter


def compute_skeleton(data: pyg.data.Data) -> pyg.data.Data:

    if "face" not in data:
        face, surface_idcs = tets_to_face(data.tets)
        pos = data.pos[surface_idcs]

        boundary_idcs = {
            key: volume_to_surface(data[f"{key}_index"], surface_idcs)
            for key in ("inlet", "outlet_ica", "outlet_eca")
        }

    else:
        pos, face = data.pos, data.face
        boundary_idcs = {
            key: data[f"{key}_index"] for key in ("inlet", "outlet_ica", "outlet_eca")
        }

    data.pos_skeleton = geodesic_rings_skeletonisation(
        pos, face, boundary_idcs, num_segments=12
    )

    return data


def tets_to_face(tets: torch.Tensor) -> tuple:

    # External and internal triangles
    face = torch.cat(
        (tets[[0, 1, 2]], tets[[0, 1, 3]], tets[[0, 2, 3]], tets[[1, 2, 3]]), dim=-1
    )

    # Internal triangles have duplicates (opposite winding)
    _, idcs, counts = torch.unique(
        torch.sort(face, dim=0)[0], return_inverse=True, return_counts=True, dim=-1
    )
    face = face[:, counts[idcs] == 1]

    # Coalesce vertices and produce volume-to-surface mapping
    surface_idcs, unique_inverse = torch.unique(face, return_inverse=True)
    face = unique_inverse.reshape(face.shape).int()

    return face, surface_idcs.int()


def volume_to_surface(
    volume_idcs: torch.Tensor, surface_idcs: torch.Tensor
) -> torch.Tensor:
    return (
        torch.nonzero(torch.any(surface_idcs[:, None] == volume_idcs[None, :], dim=-1))
        .squeeze()
        .int()
    )


def geodesic_rings_skeletonisation(
    pos: torch.Tensor, face: torch.Tensor, boundary_idcs: dict, num_segments: int
) -> torch.Tensor:

    # Compute geodesics to all boundaries
    diffusion_solver = pp3d.MeshHeatMethodDistanceSolver(pos.numpy(), face.T.numpy())
    geodesics_to = {
        key: torch.from_numpy(diffusion_solver.compute_distance_multisource(value))
        for key, value in boundary_idcs.items()
    }

    # Only the nearest boundary must contribute
    nb = {
        key: torch.prod(
            value <= torch.vstack(tuple(geodesics_to.values())), dim=0
        ).bool()
        for key, value in geodesics_to.items()
    }

    # Collapse geodesic segments into points
    geodesics_to = {key: value[nb[key]] for key, value in geodesics_to.items()}
    segment_idcs = {
        key: segment_along_geodesic(value, num_segments)
        for key, value in geodesics_to.items()
    }

    pos_skeleton = {
        key: scatter(pos[nb[key]], value, dim=0, reduce="mean")
        for key, value in segment_idcs.items()
    }

    return torch.cat(list(pos_skeleton.values()))


def segment_along_geodesic(geodesics: torch.Tensor, num_segments: int) -> torch.Tensor:
    segment_limits = torch.linspace(geodesics.min(), geodesics.max(), num_segments)

    return torch.sum(
        geodesics.view(-1, 1) < (geodesics.min() + segment_limits).view(1, -1), dim=-1
    )


def query_vectors_to_skeleton(
    pos: torch.Tensor, pos_skeleton: torch.Tensor
) -> torch.Tensor:
    _, target = knn(pos_skeleton, pos, k=2)

    pos_P = pos

    pos_A = pos_skeleton[target[0::2]]
    pos_B = pos_skeleton[target[1::2]]

    vec_AP = pos_P - pos_A
    vec_AB = pos_B - pos_A

    dist_AP = vec_AP.norm(dim=1, keepdim=True).clamp(min=1e-16)
    dist_AB = vec_AB.norm(dim=1, keepdim=True).clamp(min=1e-16)

    alpha = torch.acos((vec_AP * vec_AB).sum(dim=1, keepdim=True) / dist_AP / dist_AB)
    sine = torch.sin(torch.pi / 2.0 - alpha)

    return pos_A + dist_AP * sine * vec_AB / dist_AB - pos
