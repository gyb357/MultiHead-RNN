import torch.nn as nn
from ncps.torch import LTC, CfC


def get_rnn_layer(
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        batch_first: bool = True
) -> nn.Module:
    rnn_type = rnn_type.lower()
    types = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'ltc': LTC,
        'cfc': CfC
    }

    if rnn_type not in types:
        raise ValueError(f"Unsupported rnn type: {rnn_type}. Supported types are: {list(types.keys())}")
    
    # Standard RNNs
    if rnn_type in ['rnn', 'lstm', 'gru']:
        rnn = types[rnn_type](
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first
        )
    # Liquid Time Constant Neural Network (LTC) & Closed-form Continuous-time Neural Network (CfC)
    elif rnn_type in ['ltc', 'cfc']:
        rnn = types[rnn_type](
            input_size=input_size,
            units=hidden_size,
            batch_first=batch_first,
            # LTC specific parameters
            # ode_unfolds=1,

            # CfC specific parameters
            backbone_units=16,
            backbone_layers=1,
            activation='tanh'
        )
    return rnn


def init_weights(model: nn.Module) -> None:
    # Standard RNNs initialization
    if isinstance(model, (nn.RNN, nn.LSTM, nn.GRU)):
        for name, param in model.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def get_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

