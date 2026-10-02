import glob
import os
from pathlib import Path

import h5py
import torch
import torch_geometric
import tqdm

from data import MultiscaleData as Data


class InMemoryLumenDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        self.path_to_hdf5_file = glob.glob(os.path.join(self.root, "raw", "*.hdf5"))[0]

        with h5py.File(self.path_to_hdf5_file, "r") as hdf5_file:
            sample_ids = [
                os.path.join(self.path_to_hdf5_file, sample_id)
                for sample_id in list(hdf5_file)
            ]

        return [
            os.path.relpath(sample_id, os.path.join(self.root, "raw"))
            for sample_id in sample_ids
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        return

    def process(self):
        data_list = []

        for path in tqdm.tqdm(self.raw_paths, desc="Reading", leave=False):
            data = self.read_hdf5_data(path)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(data)
                for data in tqdm.tqdm(data_list, desc="Transforming", leave=False)
            ]

        # Save to disk
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def read_hdf5_data(path):
        hdf5_file, sample = os.path.split(path)

        with h5py.File(hdf5_file, "r") as hdf5_file:

            data = Data(
                y=torch.from_numpy(hdf5_file[sample]["velocity"][()]),
                pos=torch.from_numpy(hdf5_file[sample]["pos_tets"][()]),
                tets=torch.from_numpy(hdf5_file[sample]["tets"][()].T),
                inlet_index=torch.from_numpy(hdf5_file[sample]["inlet_idcs"][()]),
                lumen_wall_index=torch.from_numpy(
                    hdf5_file[sample]["lumen_wall_idcs"][()]
                ),
                outlets_index=torch.from_numpy(hdf5_file[sample]["outlets_idcs"][()]),
            )

        return data

    def get_geometry_for_visualisation(self, idx):

        with h5py.File(Path(self.raw_paths[idx]).parents[0], "r") as hdf5_file:

            data = torch_geometric.data.Data(
                pos=torch.from_numpy(
                    hdf5_file[Path(self.raw_paths[idx]).stem]["pos_tets"][()]
                ),
                tets=torch.from_numpy(
                    hdf5_file[Path(self.raw_paths[idx]).stem]["tets"][()].T
                ),
            )

        return data


class InMemoryAfterfifteenDataset(InMemoryLumenDataset):

    @staticmethod
    def read_hdf5_data(path):
        path_to_hdf5, sample = os.path.split(path)

        with h5py.File(path_to_hdf5, "r") as hdf5_file:

            if "4d-flow-mri" in path_to_hdf5:
                data = Data(
                    y=torch.from_numpy(hdf5_file[sample]["velocity"][()]),
                    wss=torch.from_numpy(hdf5_file[sample]["wss"][()]),
                    pos=torch.from_numpy(hdf5_file[sample]["pos_tets"][()]),
                    tets=torch.from_numpy(hdf5_file[sample]["tets"][()].T),
                    surf_normal=torch.from_numpy(
                        hdf5_file[sample]["surface_normal"][()]
                    ),  # substring "face" corrupts mini-batching
                    inlet_index=torch.from_numpy(hdf5_file[sample]["inlet_idcs"][()]),
                    lumen_wall_index=torch.from_numpy(
                        hdf5_file[sample]["lumen_wall_idcs"][()]
                    ),
                    outlets_index=torch.from_numpy(
                        hdf5_file[sample]["outlets_idcs"][()]
                    ),
                    outlet_ica_index=torch.from_numpy(
                        hdf5_file[sample]["outlet_ica_idcs"][()]
                    ),
                    outlet_eca_index=torch.from_numpy(
                        hdf5_file[sample]["outlet_eca_idcs"][()]
                    ),
                    patient_id=hdf5_file[sample].attrs["patient id"],
                )

            elif "black-blood-mri" in path_to_hdf5:
                data = Data(
                    pos=torch.from_numpy(hdf5_file[sample]["pos_face"][()]),
                    face=torch.from_numpy(hdf5_file[sample]["face"][()].T),
                    inlet_index=torch.from_numpy(hdf5_file[sample]["inlet_idcs"][()]),
                    lumen_wall_index=torch.from_numpy(
                        hdf5_file[sample]["lumen_wall_idcs"][()]
                    ),
                    outlets_index=torch.from_numpy(
                        hdf5_file[sample]["outlets_idcs"][()]
                    ),
                    outlet_ica_index=torch.from_numpy(
                        hdf5_file[sample]["outlet_ica_idcs"][()]
                    ),
                    outlet_eca_index=torch.from_numpy(
                        hdf5_file[sample]["outlet_eca_idcs"][()]
                    ),
                    patient_id=hdf5_file[sample].attrs["patient id"],
                )

            else:
                raise RuntimeError(
                    'Neither "4d-flow-mri" nor "black-blood-mri" in path to dataset.'
                )

        return data

    def get_geometry_for_visualisation(self, idx):

        with h5py.File(Path(self.raw_paths[idx]).parents[0], "r") as hdf5_file:

            if "4d-flow-mri" in self.raw_paths[idx]:
                data = torch_geometric.data.Data(
                    pos=torch.from_numpy(
                        hdf5_file[Path(self.raw_paths[idx]).stem]["pos_tets"][()]
                    ),
                    tets=torch.from_numpy(
                        hdf5_file[Path(self.raw_paths[idx]).stem]["tets"][()].T
                    ),
                )

            elif "black-blood-mri" in self.raw_paths[idx]:
                data = torch_geometric.data.Data(
                    pos=torch.from_numpy(
                        hdf5_file[Path(self.raw_paths[idx]).stem]["pos_face"][()]
                    ),
                    face=torch.from_numpy(
                        hdf5_file[Path(self.raw_paths[idx]).stem]["face"][()].T
                    ),
                )

            else:
                raise RuntimeError(
                    'Neither "4d-flow-mri" nor "black-blood-mri" in path to dataset.'
                )

        return data
