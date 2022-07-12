import torch
from .utils import build_mlp
nn = torch.nn
F = nn.functional
td = torch.distributions


class Critic(nn.Module):
    def __init__(self, in_features, layers, act=nn.ELU):
        super().__init__()
        self.qs = nn.ModuleList([build_mlp(in_features, *layers, 1, act=act) for _ in range(2)])

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        qs = [head(x) for head in self.qs]
        return torch.cat(qs, -1)


class Actor(nn.Module):
    def __init__(self, in_features, out_features, layers, act=nn.ELU):
        super().__init__()
        self.mlp = build_mlp(in_features, *layers, 2*out_features, act=act)

    def forward(self, state):
        out = self.mlp(state)
        mean, stddev = out.chunk(2, -1)
        mean = torch.tanh(mean)
        stddev = F.softplus(stddev) + 1e-4
        dist = td.transformed_distribution.TransformedDistribution(
            td.Normal(mean, stddev),
            td.IndependentTransform(td.transforms.TanhTransform(cache_size=1),
                                    reinterpreted_batch_ndims=1,
                                    cache_size=1
                                    )
        )
        return dist


class PointCloudEncoder(nn.Module):
    """PointNet with an option to process global features of selected points."""
    def __init__(self, in_channels, num_frames, out_features, layers, act=nn.ELU, features_from_layers=()):
        super().__init__()

        layers = (in_channels,) + layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            block = nn.Sequential(
                nn.Linear(layers[i], layers[i + 1]),
                act(),
            )
            self.layers.append(block)

        if isinstance(features_from_layers, int):
            features_from_layers = (features_from_layers, )
        self.selected_layers = features_from_layers

        self.fc_size = layers[-1] * (1 + sum([layers[i] for i in self.selected_layers]))
        self.ln_emb = nn.Sequential(
            nn.Linear(num_frames*self.fc_size, out_features),
            nn.LayerNorm(out_features),
            nn.Tanh()
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        values, indices = x.max(-2)
        if len(self.selected_layers):
            selected_features = torch.cat(
                [self._gather(features[ind], indices) for ind in self.selected_layers],
                -1)
            values = torch.cat((values.unsqueeze(-1), selected_features), -1).flatten(-2)
        return self.ln_emb(values.flatten(-2, -1))

    @staticmethod
    def _gather(features, indices):
        indices = torch.repeat_interleave(indices.unsqueeze(-1), features.size(-1), -1)
        return torch.gather(features, -2, indices)


class PointCloudDecoder(nn.Module):
    def __init__(self, in_features: int, pn_number: int, num_frames: int, out_channels: int, layers: tuple, act=nn.ELU):
        super().__init__()

        layers = layers + (out_channels,)
        deconvs = [
            nn.Linear(in_features, num_frames*pn_number*layers[0]),
            nn.Unflatten(-1, (num_frames, pn_number, layers[0]))
                   ]
        for i in range(len(layers)-1):
            deconvs.extend([
                act(),
                nn.Linear(layers[i], layers[i+1])
            ])

        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, x):
        return self.deconvs(x)
