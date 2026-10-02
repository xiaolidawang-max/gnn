import numpy as np
import torch
import torch_geometric as pyg
import trimesh
from torch_cluster import knn
from trimesh import Trimesh


class PointCloudSampling:
    """Creates a volumetric point cloud from a surface mesh by rejection sampling.

    Args:
        num_surface_samples (int): Number of points sampled directly from the surface.
        num_volume_samples (int): Number of points sampled from inside of the surface.
    """

    def __init__(self, num_surface_samples: int, num_volume_samples: int):
        self.num_samples = {
            "surface": num_surface_samples,
            "volume": num_volume_samples,
        }

    def __call__(self, data: pyg.data.Data):
        mesh = Trimesh(data.pos, data.face.T)
        mesh.ray = trimesh.ray.ray_pyembree.RayMeshIntersector(
            mesh, scale_to_box=False
        )  # bypass parasitary samples

        pos_surface = trimesh.sample.sample_surface_even(
            mesh, self.num_samples["surface"]
        )[0].astype("f4")
        pos_volume = trimesh.sample.volume_mesh(
            mesh, self.num_samples["volume"]
        ).astype(
            "f4"
        )  # install trimesh against pyembree

        data = self.project_idcs(data, torch.from_numpy(pos_surface))

        data.pos = torch.from_numpy(
            np.concatenate((pos_surface, pos_volume), axis=0)
        )  # indices refer to "pos_surface"
        delattr(data, "face")

        return data

    @staticmethod
    def project_idcs(data: pyg.data.Data, pos_surface: torch.Tensor):
        _, source_idcs = knn(data.pos, pos_surface, k=1)

        for key in data.keys():
            if "_index" in key:

                idcs_mask = torch.any(
                    source_idcs.view(-1, 1) == data[key].view(1, -1), dim=-1
                )
                data[key] = torch.nonzero(idcs_mask).squeeze().int()

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(num_surface_samples={self.num_samples['surface']}, num_volume_samples={self.num_samples['volume']})"
