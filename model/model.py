from . import *


class Classifier(nn.Module):
    normalizer = nn.BatchNorm1d
    activation = nn.Sigmoid

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            bias: bool = True,
            dropout: float = 0.0
    ) -> None:
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias),
            self.normalizer(hidden_size),
            self.activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes, bias=bias)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class MultiHead(nn.Module):
    def __init__(
            self,
            num_variables: int,
            rnn_hidden_size: int,
            projection_size: int,
            fc_hidden_size: int,
            num_classes: int,
            rnn_type: str,
            t_max: float,
            dropout: float = 0.0,
            **kwargs: Any
    ) -> None:
        super(MultiHead, self).__init__()
        # RNN layers
        self.rnn = nn.ModuleList([
            create_rnn_layer(
                rnn_type=rnn_type,
                input_size=1,
                hidden_size=rnn_hidden_size,
                **kwargs
            )
            for _ in range(num_variables)
        ])

        # Projection layer
        self.projection = nn.ModuleList([nn.Linear(rnn_hidden_size, projection_size) for _ in range(num_variables)])

        # Concatenated size
        concat_size = projection_size * num_variables

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(concat_size)

        # Classifier
        self.fc = Classifier(
            input_size=concat_size,
            hidden_size=fc_hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )

        # Initialize weights
        initialize_weights(self, t_max)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []

        for rnn, proj, var in zip(self.rnn, self.projection, x):
            seq, _ = rnn(var)
            outputs.append(proj(seq[:, -1, :]))

        concat = torch.concat(outputs, dim=1)
        norm = self.layer_norm(concat)

        return self.fc(norm)


class SingleHead(nn.Module):
    def __init__(
            self,
            num_variables: int,
            rnn_hidden_size: int,
            projection_size: int,
            fc_hidden_size: int,
            num_classes: int,
            rnn_type: str,
            t_max: float,
            dropout: float = 0.0,
            **kwargs: Any
    ) -> None:
        super(SingleHead, self).__init__()
        # RNN layer
        self.rnn = create_rnn_layer(
            rnn_type=rnn_type,
            input_size=num_variables,
            hidden_size=rnn_hidden_size,
            **kwargs
        )
        
        # Projection layer
        self.projection = nn.ModuleList([nn.Linear(rnn_hidden_size, projection_size) for _ in range(num_variables)])

        # Concatenated size
        concat_size = projection_size * num_variables

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(concat_size)

        # Classifier
        self.fc = Classifier(
            input_size=concat_size,
            hidden_size=fc_hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )

        # Initialize weights
        initialize_weights(self, t_max)

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=-1)

        seq, _ = self.rnn(x)

        outputs = []
        for proj in self.projection:
            outputs.append(proj(seq[:, -1, :]))

        concat = torch.cat(outputs, dim=1)
        norm = self.layer_norm(concat)

        return self.fc(norm)

