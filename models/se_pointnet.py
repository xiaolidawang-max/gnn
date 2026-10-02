import torch
from e3nn.o3 import Irreps
from torch_geometric.nn import PointNetConv

from segnn.balanced_irreps import BalancedIrreps
from segnn.batch_norm import BatchNorm
from segnn.o3_building_blocks import O3SwishGate, O3TensorProduct

from .pointnet import PointNet


def get_o3_swish_gate_irreps(irreps: Irreps):

    # See https://github.com/RobDHess/Steerable-E3-GNN/blob/main/models/segnn/o3_building_blocks.py#L152
    irreps_g_scalars = Irreps(str(irreps[0]))
    irreps_g_gate = Irreps(
        "{}x0e".format(irreps.num_irreps - irreps_g_scalars.num_irreps)
    )
    irreps_g_gated = Irreps(str(irreps[1:]))

    irreps_dict = {
        "merged": (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify(),
        "separate": (irreps_g_scalars, irreps_g_gate, irreps_g_gated),
    }

    return irreps_dict


class SEMLP(torch.nn.Module):
    def __init__(
        self,
        irreps: tuple,
        edge_attr_irreps=None,
        plain_last: bool = True,
        use_norm: bool = True,
    ):
        super().__init__()

        self.linear_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        for irreps_in, irreps_out in zip(irreps[:-2], irreps[1:-1]):
            self.linear_layers.append(
                O3TensorProduct(
                    irreps_in,
                    get_o3_swish_gate_irreps(irreps_out)["merged"],
                    edge_attr_irreps,
                )
            )
            self.norm_layers.append(
                BatchNorm(get_o3_swish_gate_irreps(irreps_out)["merged"])
                if use_norm
                else torch.nn.Identity()
            )
            self.activations.append(
                O3SwishGate(*get_o3_swish_gate_irreps(irreps_out)["separate"])
            )

        if plain_last:
            self.linear_layers.append(O3TensorProduct(*irreps[-2:], edge_attr_irreps))
            self.norm_layers.append(torch.nn.Identity())
            self.activations.append(torch.nn.Identity())
        else:
            self.linear_layers.append(
                O3TensorProduct(
                    irreps[-2],
                    get_o3_swish_gate_irreps(irreps[-1])["merged"],
                    edge_attr_irreps,
                )
            )
            self.norm_layers.append(
                BatchNorm(get_o3_swish_gate_irreps(irreps[-1])["merged"])
                if use_norm
                else torch.nn.Identity()
            )
            self.activations.append(
                O3SwishGate(*get_o3_swish_gate_irreps(irreps[-1])["separate"])
            )

    def forward(self, x: torch.Tensor, edge_attr=None):

        for linear_layer, norm_layer, activation in zip(
            self.linear_layers, self.norm_layers, self.activations
        ):
            x = activation(norm_layer(linear_layer(x, edge_attr)))

        return x


class SEPointNet(PointNet):
    def __init__(
        self, input_irreps: Irreps, output_irreps: Irreps, num_latent_channels: int
    ):
        super().__init__()

        li = [
            BalancedIrreps(lmax=1, vec_dim=vec_dim)
            for vec_dim in [num_latent_channels] * 5
        ]  # latent irreps
        vi = Irreps("1x1o")  # vector irrep

        kwargs = {"add_self_loops": False, "aggr": "mean"}
        self.sa0_conv = PointNetConv(
            SEMLP((input_irreps + vi, li[0], li[0], li[1]), use_norm=False), **kwargs
        )
        self.sa1_conv = PointNetConv(
            SEMLP((li[1] + vi, li[1], li[1], li[2]), plain_last=False), **kwargs
        )
        self.sa2_conv = PointNetConv(
            SEMLP((li[2] + vi, li[2], li[2], li[3]), plain_last=False), **kwargs
        )
        self.sa3_conv = PointNetConv(
            SEMLP((li[3] + vi, li[3], li[3], li[4]), plain_last=False), **kwargs
        )
        self.sa4_conv = PointNetConv(
            SEMLP((li[4] + vi, li[4], li[4], li[4]), plain_last=False), **kwargs
        )

        self.fp4_mlp = SEMLP((li[4] + li[4], li[4], li[4]), plain_last=False)
        self.fp3_mlp = SEMLP((li[4] + li[3], li[3], li[3]), plain_last=False)
        self.fp2_mlp = SEMLP((li[3] + li[2], li[2], li[2]), plain_last=False)
        self.fp1_mlp = SEMLP((li[2] + li[1], li[1], li[1]), plain_last=False)
        self.fp0_mlp = SEMLP(
            (li[1] + input_irreps, li[0], li[0], li[0]), use_norm=False
        )

        self.mlp = SEMLP((li[0], output_irreps))

        print(
            f"SE-PointNet++ ({sum(param.numel() for param in self.parameters() if param.requires_grad)} parameters)"
        )
