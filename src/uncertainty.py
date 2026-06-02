"""Interfaces para la parte de incertidumbre del laboratorio."""
import torch
import torch.nn as nn


def enable_dropout_during_inference(model: nn.Module) -> None:
    """Activa las capas Dropout incluso en modo eval()."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module, inputs: torch.Tensor, n_samples: int = 30
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repite n_samples predicciones con dropout activo y retorna media e incertidumbre."""
    enable_dropout_during_inference(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds.append(probs)
    stacked = torch.stack(preds)        # (n_samples, batch, n_clases)
    mean_pred = stacked.mean(dim=0)     # prediccion promedio
    uncertainty = stacked.std(dim=0)    # desviacion estandar = incertidumbre
    return mean_pred, uncertainty
