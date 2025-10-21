import torch.nn as nn
import torch
from model.utils import get_rnn_layer, init_weights
from model.module import Classifier
from typing import List
from torch import Tensor


class MultiHead(nn.Module):
    def __init__(
            self,
            num_variables: int,
            rnn_hidden_size: int,
            fc_hidden_size: int,
            proj_dim: int,
            num_classes: int,
            rnn_type: str
    ) -> None:
        super(MultiHead, self).__init__()
        # RNN layers
        self.rnn = nn.ModuleList([
            get_rnn_layer(
                rnn_type=rnn_type,
                input_size=1,
                hidden_size=rnn_hidden_size
            )
            for _ in range(num_variables)
        ])

        # Projection layer
        self.projection = nn.ModuleList([nn.Linear(rnn_hidden_size, proj_dim) for _ in range(num_variables)])
        self.layer_norm = nn.LayerNorm(num_variables * proj_dim)

        # MLP Classifier
        self.fc = Classifier(
            input_size=num_variables * proj_dim,
            hidden_size=fc_hidden_size,
            num_classes=num_classes
        )

        # Initialize RNN module weights
        self.apply(init_weights)

        # Initialize projection weights
        for proj in self.projection:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: List[Tensor] = []

        for rnn, proj, var in zip(self.rnn, self.projection, x):
            seq, _ = rnn(var)
            proj_ = proj(seq[:, -1, :])
            outputs.append(proj_)

        concat = torch.cat(outputs, dim=1)
        norm = self.layer_norm(concat)
        return self.fc(norm)


class SingleHead(nn.Module):
    def __init__(
            self,
            num_variables: int,
            rnn_hidden_size: int,
            fc_hidden_size: int,
            proj_dim: int,
            num_classes: int,
            rnn_type: str
    ) -> None:
        super(SingleHead, self).__init__()
        # RNN layers
        self.rnn = get_rnn_layer(
            rnn_type=rnn_type,
            input_size=num_variables,
            hidden_size=rnn_hidden_size
        )

        # Projection layer
        self.projection = nn.Linear(rnn_hidden_size, num_variables * proj_dim)
        self.layer_norm = nn.LayerNorm(num_variables * proj_dim)

        # MLP Classifier
        self.fc = Classifier(
            input_size=num_variables * proj_dim,
            hidden_size=fc_hidden_size,
            num_classes=num_classes
        )

        # Initialize RNN module weights
        self.apply(init_weights)

        # Initialize projection weights
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(x, dim=-1) # Last dimension concatenation
        seq, _ = self.rnn(x)
        proj = self.projection(seq[:, -1, :])
        norm = self.layer_norm(proj)
        return self.fc(norm)

