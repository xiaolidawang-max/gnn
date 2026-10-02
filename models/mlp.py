import torch
import torch_geometric as pyg


class MLP(torch.nn.Module):
    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int,
        num_layers: int,
        num_latent_channels: int,
    ):
        super().__init__()

        self.mlp = pyg.nn.models.MLP(
            in_channels=num_channels_in,
            hidden_channels=num_latent_channels,
            out_channels=num_channels_out,
            num_layers=num_layers,
        )

    def forward(self, data: pyg.data.Data) -> torch.Tensor:
        return self.mlp(data.x)
