import torch.nn as nn
from typing import Type
from torch import Tensor


class Classifier(nn.Module):
    activation: Type[nn.Module] = nn.GELU  # Sigmoid, Tanh, ReLU, GELU, ... etc.

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            bias: bool = True,
            dropout: float = 0.0
    ) -> None:
        super(Classifier, self).__init__()
        # Sequential layer
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias),
            self.activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes, bias),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:  # bias=False 가드
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

