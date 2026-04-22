"""Interfaces para la parte de incertidumbre del laboratorio."""

import torch
import torch.nn as nn


def enable_dropout_during_inference(model: nn.Module) -> None:
    """
    TODO:
    Activar capas Dropout durante inferencia para poder usar Monte Carlo Dropout.
    """

    raise NotImplementedError(
        "TODO: implementar enable_dropout_during_inference()."
    )


def mc_dropout_predict(
    model: nn.Module, inputs: torch.Tensor, n_samples: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TODO:
    Repetir varias predicciones con dropout activo y resumir:
    - media de prediccion,
    - medida de incertidumbre.
    """

    raise NotImplementedError("TODO: implementar mc_dropout_predict().")
