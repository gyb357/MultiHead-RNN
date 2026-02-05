from . import *


# Default arguments for RNN (RNN, LSTM, GRU)
_DEFAULT_ARGS = {
    'num_layers': 1,
    'bias': True,
    'dropout': 0.0,
    'bidirectional': False
}

# Forbidden arguments for RNN (RNN, LSTM, GRU)
_FORBIDDEN_ARGS = set(_DEFAULT_ARGS.keys())

# Allowed additional arguments
_ALLOWED_ARGS = {
    'rnn': None,
    'lstm': None,
    'gru': None,
    'ltc': {'ode_unfolds'},
    'cfc': {'activation', 'backbone_units', 'backbone_layers', 'backbone_dropout'}
}


def _validate_kwargs(
        rnn_type: str,
        kwargs: Dict[str, Any],
        forbidden: Set[str],
        allowed: Optional[Set[str]]
) -> None:
    # Extract keys from kwargs
    keys = set(kwargs.keys())

    # Check forbidden arguments
    forbidden_used = keys & forbidden
    if forbidden_used:
        raise ValueError(
            f"{sorted(forbidden_used)} are not accepted. "
            f"They are fixed to default values."
        )
    
    # Check allowed arguments
    if allowed is None:
        if keys:
            raise ValueError(
                f"'{rnn_type}' does not accept additional arguments. "
                f"Got: {sorted(keys)}."
            )
    else:
        invalid = keys - allowed
        if invalid:
            raise ValueError(
                f"Invalid arguments for '{rnn_type}': {sorted(invalid)}. "
                f"Supported arguments are: {sorted(allowed)}."
            )


def create_rnn_layer(
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        batch_first: bool = True,
        **kwargs: Any
) -> nn.Module:
    # Define rnn types
    rnn_type = rnn_type.lower()
    rnn_types = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'ltc': LTC,
        'cfc': CfC
    }
    if rnn_type not in rnn_types:
        raise ValueError(
            f"Unsupported RNN type '{rnn_type}'. "
            f"Supported types are: {sorted(rnn_types.keys())}."
        )
    
    # Validate kwargs
    _validate_kwargs(
        rnn_type,
        kwargs,
        _FORBIDDEN_ARGS,
        _ALLOWED_ARGS[rnn_type]
    )

    # Common arguments
    args = {
        'input_size': input_size,
        'batch_first': batch_first
    }

    # Specific handling for different RNN types
    if rnn_type in {'rnn', 'lstm', 'gru'}:
        args['hidden_size'] = hidden_size
        args.update(_DEFAULT_ARGS)
    else:
        args['units'] = hidden_size

    args.update(kwargs)
    
    return rnn_types[rnn_type](**args)


@torch.no_grad()
def initialize_weights(
    model: nn.Module,
    t_max: float,
    t_min: float = 2.0,
    eps: float = 1e-8
) -> None:
    cache = {}

    for m in model.modules():
        # RNN
        if isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
            for name, p in m.named_parameters(recurse=False):
                # Weights
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p)

                # Biases
                elif 'bias' in name:
                    nn.init.zeros_(p)

                    # Extract layer key for cache
                    if 'bias_ih_' in name:
                        layer_key = name.split('bias_ih_', 1)[1]
                    elif 'bias_hh_' in name:
                        layer_key = name.split('bias_hh_', 1)[1]
                    else:
                        layer_key = name

                    # PyTorch adds bias_ih and bias_hh, scale by 0.5
                    scale = 0.5 if ('bias_ih_' in name or 'bias_hh_' in name) else 1.0

                    # Chrono initialization: https://arxiv.org/abs/1804.11188
                    # LSTM
                    if isinstance(m, nn.LSTM):
                        h = p.shape[0] // 4
                        key = (id(m), layer_key, 'lstm')

                        if key not in cache:
                            t = torch.empty(h, device=p.device, dtype=p.dtype).uniform_(t_min, t_max)
                            t = t.clamp_min(1.0 + eps)

                            bf = torch.log(t - 1.0)
                            bi = -bf
                            cache[key] = (bi, bf)

                        bi, bf = cache[key]

                        # LSTM gate order: (input, forget, cell, output)
                        p[0 * h : 1 * h].copy_(bi * scale)  # input gate
                        p[1 * h : 2 * h].copy_(bf * scale)  # forget gate

                    # GRU
                    elif isinstance(m, nn.GRU):
                        h = p.shape[0] // 3
                        key = (id(m), layer_key, 'gru')

                        if key not in cache:
                            t = torch.empty(h, device=p.device, dtype=p.dtype).uniform_(t_min, t_max)
                            t = t.clamp_min(1.0 + eps)

                            bz = torch.log(t - 1.0)
                            cache[key] = bz

                        bz = cache[key]

                        # GRU gate order: (reset, update, new)
                        p[1 * h : 2 * h].copy_(bz * scale)  # update gate

        # Linear
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Normalization
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

