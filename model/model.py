import torch.nn as nn
import torch
from model.utils import get_rnn_layer, init_weights
from model.module import Classifier
from typing import List
from torch import Tensor


class MultiHead(nn.Module):
    proj_dim: int = 1

    def __init__(
            self,
            num_variables: int,
            rnn_hidden_size: int,
            fc_hidden_size: int,
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
        self.proj = nn.ModuleList([nn.Linear(rnn_hidden_size, self.proj_dim) for _ in range(num_variables)])
        for layer in self.proj:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # MLP Classifier
        self.fc = Classifier(
            input_size=num_variables,
            hidden_size=fc_hidden_size,
            num_classes=num_classes
        )

        # Initialize RNN module weights
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: List[Tensor] = []

        for rnn, proj, var in zip(self.rnn, self.proj, x):
            seq, _ = rnn(var)
            proj = proj(seq[:, -1, :])
            outputs.append(proj)

        concat = torch.cat(outputs, dim=1)
        return self.fc(concat)


class SingleHead(nn.Module):
    def __init__(
            self,
            num_variables: int,
            rnn_hidden_size: int,
            fc_hidden_size: int,
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
        self.proj = nn.Linear(rnn_hidden_size, num_variables)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # MLP Classifier
        self.fc = Classifier(
            input_size=num_variables,
            hidden_size=fc_hidden_size,
            num_classes=num_classes
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(x, dim=-1) # Last dimension concatenation
        seq, _ = self.rnn(x)
        proj = self.proj(seq[:, -1, :])
        return self.fc(proj)

