import os
import statistics
import sys
from argparse import ArgumentParser
from functools import partial
from time import asctime
from uuid import uuid4

import geomloss
import torch
import torch_geometric as pyg
import trimesh
from e3nn import o3
from e3nn.o3 import Irreps
from torch_cluster import knn, radius
from torch_geometric.transforms import Compose
from torch_scatter import scatter
from tqdm import tqdm

import wandb_impostor as wandb
from datasets import InMemoryAfterfifteenDataset
from models import MLP, EmptyPointNet, PointNet, SEPointNet
from transforms import PointCloudSampling, PointNetSampling, compute_skeleton
from transforms.fem import compute_jacobian, compute_laplacian
from transforms.skeleton import query_vectors_to_skeleton
from utils import AccuracyAnalysis, VTUWriter

parser = ArgumentParser()

parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=0)
parser.add_argument("--dir", type=str, default=f"id-{uuid4().hex}")
parser.add_argument("--wandb", type=bool, default=False)
args = parser.parse_args()

if args.dir:
    os.makedirs(args.dir, exist_ok=True)

if args.wandb:
    import wandb
else:
    import wandb_impostor as wandb

wandb_config = {
    "cross_validation_fold_idx": 0,
    "batch_size": 14,
    "model": "pointnet",
    "learning_rate": 8e-4,
    "lr_decay_gamma": 0.9955,
    "num_epochs": args.num_epochs,
    "loss_term_factors": {
        "generic": 1.0,
        "dirichlet_bct": None,
        "neumann_bct": None,
        "continuity": 0.0,
        "momentum": 0.0,
        "approximation": 0.0,
    },
}

cross_validation_fold_size = 28

rho = 1060  # density [kg/m^3]
mu = 0.004  # dynamic viscosity [kg/m/s]

luxury_log_frequency = 100  # currently unused


def main(rank, num_gpus):
    ddp_setup(rank, num_gpus)

    afterfifteen_dataset = InMemoryAfterfifteenDataset(
        "afterfifteen-dataset/4d-flow-mri",
        pre_transform=Compose(
            [
                PointNetSampling(
                    rel_ratios=(0.2, 0.25, 0.25, 0.25, 0.25),
                    edge_radii=(0.0012, 0.0020, 0.0035, 0.0071, 0.0224),
                    dim_simplex=3,
                ),
                compute_skeleton,
                geometric_input_transform,
            ]
        ),
    )

    # n-fold cross validation
    training_dataset_index, first_test_sample_idx, last_test_sample_idx = (
        get_cross_validation_split(
            num_samples=len(afterfifteen_dataset),
            fold_size=cross_validation_fold_size,
            fold_idx=wandb.config["cross_validation_fold_idx"],
        )
    )

    training_data_loader = pyg.loader.DataLoader(
        afterfifteen_dataset[[training_dataset_index][rank]],
        batch_size=wandb.config["batch_size"],
        shuffle=True,
    )
    validation_data_loader = pyg.loader.DataLoader(
        afterfifteen_dataset[
            [slice(first_test_sample_idx, last_test_sample_idx)][rank]
        ],
        batch_size=1,
        shuffle=False,
    )
    test_dataset_slice = slice(first_test_sample_idx, last_test_sample_idx)
    visualisation_dataset_range = range(first_test_sample_idx, last_test_sample_idx)

    match wandb.config["model"]:
        case "pointnet":
            neural_network = PointNet(
                num_channels_in=27, num_channels_out=3, num_latent_channels=184
            )
        case "se_pointnet":
            neural_network = SEPointNet(
                input_irreps=Irreps("6x1o+9x0e"),
                output_irreps=Irreps("1x1o"),
                num_latent_channels=308,
            )

        case "mlp":
            neural_network = MLP(
                num_channels_in=27,
                num_channels_out=3,
                num_layers=8,
                num_latent_channels=405,
            )  # 1004808 parameters
        case "empty_pointnet":
            neural_network = EmptyPointNet(
                num_channels_in=2, num_channels_out=3, num_latent_channels=184
            )

    training_device = torch.device(f"cuda:{rank}")
    neural_network.to(training_device)

    load_neural_network_weights(neural_network)

    # Distributed data parallel (multi-GPU training)
    neural_network = ddp_module(neural_network, rank)

    wandb.watch(neural_network)
    optimisation_loop(
        rank,
        {
            "neural_network": neural_network,
            "training_device": training_device,
            "training_data_loader": training_data_loader,
            "validation_data_loader": validation_data_loader,
        },
    )

    ddp_rank_zero(
        assessment_loop,
        {
            "neural_network": neural_network,
            "training_device": training_device,
            "dataset": afterfifteen_dataset,
            "test_dataset_slice": test_dataset_slice,
            "visualisation_dataset_range": visualisation_dataset_range,
        },
    )

    ddp_cleanup()


@torch.no_grad()
def geometric_input_transform(data):
    side_encoding = torch.tensor(
        {"l": 0, "r": 1}[data["patient_id"][-1]], device=data.pos.device
    )

    vectors_to = {
        key: data.pos[value.long()] - data.pos
        for key, value in compute_nearest_boundary_vertex(data).items()
    }
    vectors_to["centerline"] = query_vectors_to_skeleton(data.pos, data.pos_skeleton)
    distances_to = {
        key: torch.linalg.norm(value, dim=-1, keepdim=True)
        for key, value in vectors_to.items()
    }

    keys = ("inlet", "lumen_wall", "outlets", "outlet_ica", "outlet_eca", "centerline")
    data.x = torch.cat(
        (
            *(
                vectors_to[key] / torch.clamp(distances_to[key], min=1e-16)
                for key in keys
            ),
            *(distances_to[key] * 1e2 for key in keys),  # [m] to [cm]
            side_encoding.float().expand(data.pos.size(0), 1),
        ),
        dim=-1,
    )

    return data


def compute_nearest_boundary_vertex(data):
    index_dict = {}

    for key in data.keys():
        if "inlet" in key or "lumen_wall" in key or "outlet" in key:
            index_dict[key.replace("_index", "")] = data[f"{key}"][
                knn(data.pos[data[f"{key}"].long()], data.pos, k=1)[1].long()
            ]

    return index_dict


def get_cross_validation_split(num_samples, fold_size, fold_idx):
    first_test_sample_idx = fold_size * fold_idx

    training_dataset_index = torch.ones(num_samples, dtype=torch.bool)
    training_dataset_index[
        slice(first_test_sample_idx, first_test_sample_idx + fold_size)
    ] = False

    last_test_sample_idx = first_test_sample_idx + fold_size
    last_test_sample_idx = (
        last_test_sample_idx if last_test_sample_idx < num_samples else num_samples
    )

    return training_dataset_index, first_test_sample_idx, last_test_sample_idx


def load_neural_network_weights(neural_network):
    if os.path.exists(os.path.join(args.dir, "neural_network_weights.pt")):

        neural_network.load_state_dict(
            torch.load(os.path.join(args.dir, "neural_network_weights.pt"))
        )
        print("Resuming from pre-trained neural-network weights.")


def optimisation_loop(rank, config):

    loss_function = torch.nn.L1Loss()

    optimiser = torch.optim.Adam(
        config["neural_network"].parameters(), lr=wandb.config["learning_rate"]
    )
    load_optimiser_state(rank, optimiser)

    learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimiser, gamma=wandb.config["lr_decay_gamma"]
    )

    for epoch in tqdm(
        range(wandb.config["num_epochs"]), desc="Epochs", position=0, leave=True
    ):

        loss_data = {
            key: {
                "generic": [],
                # "dirichlet_bct": [],
                # "neumann_bct": [],
                "continuity": [],
                "momentum": [],
                "approximation": [],
            }
            for key in ("training", "validation")
        }

        # Objective convergence
        config["neural_network"].train()

        for batch in tqdm(
            config["training_data_loader"],
            desc="Training split",
            position=1,
            leave=False,
        ):
            optimiser.zero_grad()

            batch = batch.to(config["training_device"])
            prediction = config["neural_network"](
                input_transform(batch, velocity_field=batch.y)
            )

            jacobian = compute_jacobian(batch.pos, batch.tets, prediction)

            loss_term_factors = wandb.config["loss_term_factors"]
            loss_terms = {
                "generic": loss_function(prediction, batch.y),
                # "dirichlet_bct": torch.mean(
                #     torch.linalg.norm(prediction[batch.lumen_wall_index], dim=-1)
                # ),
                # "neumann_bct": loss_function(
                #     compute_wss(jacobian, batch.surf_normal), batch.wss
                # ),
                "continuity": torch.mean(
                    torch.abs(compute_continuity_residual(jacobian))
                ),
                "momentum": torch.mean(
                    torch.linalg.norm(
                        compute_momentum_residual(batch, jacobian, prediction), dim=-1
                    )
                ),
                "approximation": torch.mean(
                    compute_approximation_error(prediction, batch.y, batch.batch)
                ),
            }
            loss_value = sum(
                (loss_term_factors[key] * value for key, value in loss_terms.items())
            )

            # if epoch % luxury_log_frequency == 0:
            #     reference_field, field = local_balls_averages_fields(
            #         reference_pos=batch.pos,
            #         reference_field=prediction,
            #         pos=batch.pos,
            #         field=batch.y,
            #         reference_batch=batch.batch,
            #         batch=batch.batch,
            #     )

            #     loss_data["training"].setdefault("delta", [])
            #     loss_terms["delta"] = (field - reference_field).norm(dim=-1).mean()

            for key, value in loss_terms.items():
                loss_data["training"][key].append(value.item())

            loss_value.backward()  # "autograd" hook fires and triggers gradient synchronisation across processes
            torch.nn.utils.clip_grad_norm_(
                config["neural_network"].parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )

            optimiser.step()

            del batch, prediction

        learning_rate_scheduler.step()

        ddp_rank_zero(
            torch.save,
            config["neural_network"].state_dict(),
            os.path.join(args.dir, "neural_network_weights.pt"),
        )
        torch.save(
            optimiser.state_dict(),
            os.path.join(args.dir, f"rank_{rank}_optimiser_state.pt"),
        )

        # Learning task
        # config['neural_network'].eval()  # training-mode "BatchNorm" approximates "instance norm"

        with torch.no_grad():
            for batch in tqdm(
                config["validation_data_loader"],
                desc="Validation split",
                position=1,
                leave=False,
            ):

                batch = batch.to(config["training_device"])
                prediction = config["neural_network"](
                    input_transform(batch, velocity_field=batch.y)
                )

                jacobian = compute_jacobian(batch.pos, batch.tets, prediction)

                loss_terms = {
                    "generic": loss_function(prediction, batch.y),
                    # "dirichlet_bct": torch.mean(
                    #     torch.linalg.norm(prediction[batch.lumen_wall_index], dim=-1)
                    # ),
                    # "neumann_bct": loss_function(
                    #     compute_wss(jacobian, batch.surf_normal), batch.wss
                    # ),
                    "continuity": torch.mean(
                        torch.abs(compute_continuity_residual(jacobian))
                    ),
                    "momentum": torch.mean(
                        torch.linalg.norm(
                            compute_momentum_residual(batch, jacobian, prediction),
                            dim=-1,
                        )
                    ),
                    "approximation": torch.mean(
                        compute_approximation_error(prediction, batch.y, batch.batch)
                    ),
                }

                # if epoch % luxury_log_frequency == 0:
                #     reference_field, field = local_balls_averages_fields(
                #         reference_pos=batch.pos,
                #         reference_field=prediction,
                #         pos=batch.pos,
                #         field=batch.y,
                #         reference_batch=batch.batch,
                #         batch=batch.batch,
                #     )

                #     loss_data["validation"].setdefault("delta", [])
                #     loss_terms["delta"] = (field - reference_field).norm(dim=-1).mean()

                #     loss = local_balls_sinkhorn_loss(
                #         batch.pos,
                #         prediction,
                #         batch.pos,
                #         (batch.y,),
                #         reference_batch=batch.batch,
                #         batch=batch.batch,
                #     )[0]

                #     loss_data["validation"].setdefault("wasserstein", [])
                #     loss_terms["wasserstein"] = loss.mean()

                #     cosine_similarity = torch.nn.CosineSimilarity()(batch.y, prediction)
                #     wandb.log({f'{batch.patient_id[0]}': {
                #         'magnitude': field_to_wandb(batch.pos[batch.batch == 0], (prediction - batch.y)[batch.batch == 0].norm(dim=-1)),
                #         'direction': field_to_wandb(batch.pos[batch.batch == 0], 1. - cosine_similarity[batch.batch == 0])
                #     }})

                for key, value in loss_terms.items():
                    loss_data["validation"][key].append(value.item())

                del batch, prediction

        for phase_name in loss_data.keys():
            wandb.log(
                {
                    phase_name: {
                        key: statistics.mean(value)
                        for key, value in loss_data[phase_name].items()
                    }
                }
            )


def load_optimiser_state(rank, optimiser):
    if os.path.exists(os.path.join(args.dir, f"rank_{rank}_optimiser_state.pt")):

        optimiser.load_state_dict(
            torch.load(os.path.join(args.dir, f"rank_{rank}_optimiser_state.pt"))
        )
        print("Resuming from previous optimiser state.")


@torch.no_grad()
def input_transform(data, velocity_field):

    data.x = torch.cat(
        (
            data.x,
            torch.linalg.norm(
                torch.mean(velocity_field[data["inlet_index"]], dim=0)
            ).expand(data.pos.size(0), 1),
            torch.linalg.norm(
                torch.std(velocity_field[data["inlet_index"]], dim=0)
            ).expand(data.pos.size(0), 1),
        ),
        dim=-1,
    )

    return data


def compute_wss(jacobian, surface_normal):
    return mu * project_onto_plane(
        (jacobian @ surface_normal[..., None]).squeeze(), surface_normal
    )


def project_onto_plane(vector, normal):
    normal = normal / torch.clip(
        torch.linalg.norm(normal, dim=-1, keepdims=True), min=1e-16, max=None
    )

    return vector - torch.sum(vector * normal, dim=-1, keepdims=True) * normal


def compute_continuity_residual(jacobian):
    return torch.sum(jacobian.diagonal(dim1=1, dim2=2), dim=-1)


def compute_momentum_residual(data, jacobian, velocity_field):
    laplacian = compute_laplacian(data.pos, data.tets, jacobian)

    return (
        rho * (jacobian @ velocity_field[..., None]).squeeze() - mu * laplacian
    )  # steady flow, pressure gradient (ε)


def compute_approximation_error(prediction, ground_truth, batch):

    numerator = scatter(
        (ground_truth - prediction).norm(dim=-1) ** 2, batch, dim=0, reduce="sum"
    )
    denominator = scatter(ground_truth.norm(dim=-1) ** 2, batch, dim=0, reduce="sum")

    return torch.sqrt(numerator / denominator)


def local_balls_averages_fields(
    reference_pos,
    reference_field,
    pos,
    field,
    balls_radius=0.0025,
    reference_batch=None,
    batch=None,
):

    source_idcs, target_idcs = radius(
        reference_pos,
        reference_pos,
        r=balls_radius,
        batch_x=reference_batch,
        batch_y=reference_batch,
        max_num_neighbors=256,
    )
    reference_field = scatter(
        reference_field[target_idcs], source_idcs, dim=0, reduce="mean"
    )

    source_idcs, target_idcs = radius(
        pos,
        reference_pos,
        r=balls_radius,
        batch_x=batch,
        batch_y=reference_batch,
        max_num_neighbors=256,
    )
    field = scatter(field[target_idcs], source_idcs, dim=0, reduce="mean")

    return reference_field, field


def field_to_wandb(pos, field):
    pos = (pos := pos - pos.mean(dim=0)) / pos.abs().max()
    field = field.view(-1, 1) / field.max()

    return wandb.Object3D(
        torch.cat(
            (pos, torch.full_like(field, 255.0), *[255.0 * (1.0 - field)] * 2), dim=1
        )
        .cpu()
        .numpy()
    )


def assessment_loop(config):
    accuracy_analysis = {
        "4d_flow": {
            key: AccuracyAnalysis()
            for key in ("velocity", "continuity", "momentum", "distribution")
        },
        "black_blood": {
            key: AccuracyAnalysis()
            for key in ("continuity", "momentum", "velocity", "estimate")
        },
    }
    vtu_writer = VTUWriter()

    bbm_dataset = InMemoryAfterfifteenDataset(
        "afterfifteen-dataset/black-blood-mri",
        pre_transform=Compose(
            [
                compute_skeleton,  # point cloud sampling removes triangles
                PointCloudSampling(num_surface_samples=3600, num_volume_samples=18714),
                PointNetSampling(
                    rel_ratios=(0.2, 0.25, 0.25, 0.25, 0.25),
                    edge_radii=(0.0012, 0.0020, 0.0035, 0.0071, 0.0224),
                    dim_simplex=3,
                ),
                # PointCloudPoolingScales(rel_sampling_ratios=(0.2,), interp_simplex='tetrahedron'),
                geometric_input_transform,
            ]
        ),
    )

    # config['neural_network'].eval()  # training-mode "BatchNorm" equals "instance norm" (batch size one)

    with torch.no_grad():

        i = j = 0

        # Quantitative
        for data in tqdm(
            config["dataset"][config["test_dataset_slice"]],
            desc="Test split",
            position=0,
            leave=False,
        ):
            data = data.to(config["training_device"])

            # Exclude subjects for which data leaks across splits
            if data.patient_id in ("247-l", "247-r", "359-l", "359-r"):
                continue

            _input_transform = partial(input_transform, velocity_field=data.y)

            # 4D flow MRI
            prediction = config["neural_network"](_input_transform(data))
            jacobian = compute_jacobian(data.pos, data.tets, prediction)

            accuracy_analysis["4d_flow"]["velocity"].append_values(
                {
                    "ground_truth": data.y.cpu(),
                    "prediction": prediction.cpu(),
                    "scatter_idx": torch.tensor(i),
                }
            )

            accuracy_analysis["4d_flow"]["continuity"].append_values(
                {
                    "prediction": compute_continuity_residual(jacobian.cpu()).view(
                        -1, 1
                    ),
                    "scatter_idx": torch.tensor(i),
                }
            )

            accuracy_analysis["4d_flow"]["momentum"].append_values(
                {
                    "prediction": compute_momentum_residual(
                        data.clone().cpu(), jacobian.cpu(), prediction.cpu()
                    ),
                    "scatter_idx": torch.tensor(i),
                }
            )

            # Local balls Wasserstein metric
            ground_truth_loss = local_balls_sinkhorn_loss(
                data.pos, prediction, data.pos, (data.y,)
            )[0]

            accuracy_analysis["4d_flow"]["distribution"].append_values(
                {
                    "prediction": ground_truth_loss.view(-1, 1).cpu() * 1e3,
                    "scatter_idx": torch.tensor(i),
                }
            )

            # Save visualisation of Wasserstein metric
            # (pyvista_poly_data := pyvista.PolyData(data.pos.cpu().numpy()))['OT'] = ground_truth_loss.cpu().numpy()
            # pyvista_poly_data.save(os.path.join(args.dir, f"wasserstein_metric_4d_flow_mri_{data['patient_id']}.vtk"))

            # Black blood MRI
            if data["patient_id"] in bbm_dataset.patient_id:
                bbm_data = bbm_dataset.get(
                    bbm_dataset.patient_id.index(data["patient_id"])
                ).to(config["training_device"])
                bbm_data.tets = "none"

                bbm_data = align(
                    bbm_data,
                    pos_source=bbm_data.pos_skeleton,
                    pos_target=data.pos_skeleton,
                )

                bbm_prediction = config["neural_network"](_input_transform(bbm_data))
                bbm_jacobian = compute_jacobian(
                    bbm_data.pos, bbm_data.tets, bbm_prediction
                )

                accuracy_analysis["black_blood"]["continuity"].append_values(
                    {
                        "prediction": compute_continuity_residual(
                            bbm_jacobian.cpu()
                        ).view(-1, 1),
                        "scatter_idx": torch.tensor(j),
                    }
                )

                accuracy_analysis["black_blood"]["momentum"].append_values(
                    {
                        "prediction": compute_momentum_residual(
                            bbm_data.clone().cpu(),
                            bbm_jacobian.cpu(),
                            bbm_prediction.cpu(),
                        ),
                        "scatter_idx": torch.tensor(j),
                    }
                )

                # Local balls Wasserstein metric
                ground_truth_loss, prediction_loss = local_balls_sinkhorn_loss(
                    bbm_data.pos, bbm_prediction, data.pos, (data.y, prediction)
                )

                accuracy_analysis["black_blood"]["velocity"].append_values(
                    {
                        "prediction": ground_truth_loss.view(-1, 1).cpu() * 1e3,
                        "scatter_idx": torch.tensor(j),
                    }
                )

                accuracy_analysis["black_blood"]["estimate"].append_values(
                    {
                        "prediction": prediction_loss.view(-1, 1).cpu() * 1e3,
                        "scatter_idx": torch.tensor(j),
                    }
                )

                # Save visualisation of Wasserstein metric
                # (pyvista_poly_data := pyvista.PolyData(bbm_data.pos.cpu().numpy()))['OT'] = ground_truth_loss.cpu().numpy()
                # pyvista_poly_data['OT_'] = prediction_loss.cpu().numpy()
                # pyvista_poly_data.save(os.path.join(args.dir, f"wasserstein_metric_black_blood_mri_{data['patient_id']}.vtk"))

                j += 1

                del bbm_data

            i += 1

            del data

        print(
            f"4D flow MRI\nVelocity\n{accuracy_analysis['4d_flow']['velocity'].accuracy_table()}"
        )
        print(
            f"Continuity\n{accuracy_analysis['4d_flow']['continuity'].residual_table()}"
        )
        print(f"Momentum\n{accuracy_analysis['4d_flow']['momentum'].residual_table()}")
        print(
            f"Distribution\n{accuracy_analysis['4d_flow']['distribution'].residual_table()}"
        )

        wandb.log(
            {
                "test": {
                    "approximation": accuracy_analysis["4d_flow"]["velocity"]
                    .get_approximation_error()
                    .mean(),
                    **{
                        key: accuracy_analysis["4d_flow"][key].get_residual_mae().mean()
                        for key in ("continuity", "momentum")
                    },
                }
            }
        )

        print(
            f"Black blood MRI\nContinuity\n{accuracy_analysis['black_blood']['continuity'].residual_table()}"
        )
        print(
            f"Momentum\n{accuracy_analysis['black_blood']['momentum'].residual_table()}"
        )
        print(
            f"Velocity\n{accuracy_analysis['black_blood']['velocity'].residual_table()}"
        )
        print(
            f"Estimate\n{accuracy_analysis['black_blood']['estimate'].residual_table()}"
        )

        # Qualitative (visual)
        # config['neural_network'].cpu()  # avoid memory issues

        for idx in tqdm(
            config["visualisation_dataset_range"],
            desc="Visualisation split",
            position=0,
            leave=False,
        ):
            data = config["dataset"].get(idx).to(config["training_device"])

            # Exclude subjects for which data leaks across splits
            if data.patient_id in ("247-l", "247-r", "359-l", "359-r"):
                continue

            patient_id = data.pop("patient_id")
            _input_transform = partial(input_transform, velocity_field=data.y)

            # 4D flow MRI
            vtu_writer(
                os.path.join(args.dir, f"mesh_4d_flow_mri_{patient_id}.vtu"),
                config["dataset"].get_geometry_for_visualisation(idx),
            )

            data["Y"] = config["neural_network"](_input_transform(data))
            jacobian = compute_jacobian(data.pos, data.tets, data["Y"])
            data["CON"] = compute_continuity_residual(jacobian)
            data["MOM"] = compute_momentum_residual(data, jacobian, data["Y"])

            data["WSS"] = compute_wss(jacobian, data.surf_normal)

            for key in ("tets", "x"):
                data.pop(key)

            for key in data.keys():
                if "scale" in key:
                    data.pop(key)

            data.face = torch.arange(data.pos.size(0)).expand(
                3, -1
            )  # abuse VTU format to store point clouds
            vtu_writer(
                os.path.join(args.dir, f"point_cloud_4d_flow_mri_{patient_id}.vtu"),
                data.cpu(),
            )

            # Black blood MRI
            if patient_id in bbm_dataset.patient_id:
                idx = bbm_dataset.patient_id.index(patient_id)
                data = bbm_dataset.get(idx).to(config["training_device"])

                vtu_writer(
                    os.path.join(args.dir, f"mesh_black_blood_mri_{patient_id}.vtu"),
                    bbm_dataset.get_geometry_for_visualisation(idx),
                )

                data["Y"] = config["neural_network"](_input_transform(data))

                for key in ("patient_id", "x"):
                    data.pop(key)

                for key in data.keys():
                    if "scale" in key:
                        data.pop(key)

                data.face = torch.arange(data.pos.size(0)).expand(
                    3, -1
                )  # abuse VTU format to store point clouds
                vtu_writer(
                    os.path.join(
                        args.dir, f"point_cloud_black_blood_mri_{patient_id}.vtu"
                    ),
                    data.cpu(),
                )


def align(data, pos_source, pos_target):

    rototransflection_map = torch.from_numpy(
        trimesh.registration.procrustes(
            pos_source.cpu(),
            pos_target.cpu(),
            reflection=False,
            scale=False,
            return_cost=False,
        ).astype("f4")
    )

    for key in ("pos", "pos_skeleton"):
        data[key] = torch.from_numpy(
            trimesh.transformations.transform_points(
                data[key].cpu(), rototransflection_map
            ).astype("f4")
        ).to(device=data[key].device)

    data.x = data.x @ o3.Irreps("6x1o+7x0e").D_from_matrix(
        rototransflection_map[:3, :3]
    ).T.to(device=data.x.device)

    return data


@torch.no_grad()  # this is necessary for some reason
def local_balls_sinkhorn_loss(
    reference_pos,
    reference_field,
    pos,
    fields,
    balls_radius=0.0025,
    reference_batch=None,
    batch=None,
):

    reference_source_idcs, reference_target_idcs = radius(
        reference_pos,
        reference_pos,
        r=balls_radius,
        batch_x=reference_batch,
        batch_y=reference_batch,
        max_num_neighbors=256,
    )
    reference_weight, reference_field = dummy_batch_local_balls(
        reference_source_idcs,
        reference_target_idcs,
        torch.cat((reference_pos, reference_field), dim=-1),
    )

    source_idcs, target_idcs = radius(
        pos,
        reference_pos,
        r=balls_radius,
        batch_x=batch,
        batch_y=reference_batch,
        max_num_neighbors=256,
    )
    weights_and_fields = (
        dummy_batch_local_balls(
            source_idcs, target_idcs, torch.cat((pos, field), dim=-1)
        )
        for field in fields
    )

    loss_function = geomloss.SamplesLoss("sinkhorn", blur=1e-5)

    return [
        loss_function(reference_weight, reference_field, weight, field)
        for weight, field in weights_and_fields
    ]  # must be a list


def dummy_batch_local_balls(source_idcs, target_idcs, field):
    unique_source_idcs, num_vectors = source_idcs.unique(return_counts=True)

    # allow empty neighbourhoods
    dummy_num_vectors = torch.zeros(
        source_idcs.max() + 1, dtype=num_vectors.dtype, device=num_vectors.device
    )
    dummy_num_vectors[unique_source_idcs] = num_vectors
    num_vectors = dummy_num_vectors

    max_num_vectors = num_vectors.max()

    batch_index = [
        torch.nn.functional.pad(
            torch.ones(num, dtype=torch.bool), (0, max_num_vectors - num)
        )
        for num in num_vectors
    ]
    batch_index = torch.cat(batch_index).to(field.device)

    dummy_batch = torch.zeros(
        (max_num_vectors * num_vectors.numel(), field.size(-1)), device=field.device
    )
    dummy_batch[batch_index] = field[target_idcs]
    dummy_batch = dummy_batch.view(-1, max_num_vectors, field.size(-1))

    weight = batch_index.float().view(-1, max_num_vectors)

    weight[weight.sum(dim=1) == 0.0] = 1.0  # ε to avoid exploding transport cost
    weight = weight / weight.sum(dim=1, keepdim=True)

    return weight, dummy_batch


def ddp_setup(rank, num_gpus):

    if num_gpus > 1:

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        sys.stderr = open(f"{rank}.out", "w")  # used by "tqdm"

        torch.distributed.init_process_group("nccl", rank=rank, world_size=num_gpus)
        wandb.init(
            project="afterfifteen",
            config=wandb_config,
            group=f"{args.dir or 'DDP'} ({asctime()})",
        )

    else:
        wandb.init(project="afterfifteen", config=wandb_config, name=args.dir or None)


def ddp_module(torch_module, rank):
    return (
        DistributedDataParallel(torch_module, device_ids=[rank])
        if torch.distributed.is_initialized()
        else torch_module
    )


def ddp_rank_zero(fun, *args):

    if torch.distributed.is_initialized():
        fun(*args) if torch.distributed.get_rank() == 0 else None

    else:
        fun(*args)


def ddp_cleanup():

    wandb.finish()
    (
        torch.distributed.destroy_process_group()
        if torch.distributed.is_initialized()
        else None
    )

    (
        sys.stderr.close() if torch.distributed.is_initialized() else None
    )  # last executed statement


def ddp(fun, num_gpus):
    (
        torch.multiprocessing.spawn(fun, args=(num_gpus,), nprocs=num_gpus, join=True)
        if num_gpus > 1
        else fun(rank=0, num_gpus=num_gpus)
    )


if __name__ == "__main__":
    ddp(main, args.num_gpus)
