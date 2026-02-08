import torch.nn as nn
from typing import Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from model.model import MultiHead, SingleHead


def get_scaler(name: str) -> Union[StandardScaler, RobustScaler, MinMaxScaler]:
    scalers = {
        'standard': StandardScaler,
        'robust': RobustScaler,
        'minmax': MinMaxScaler
    }

    return scalers[name.lower()]()


def get_model_class(name: str) -> Union[MultiHead, SingleHead]:
    models = {
        'multi': MultiHead,
        'single': SingleHead
    }

    return models[name.lower()]


def get_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

