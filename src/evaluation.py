"""Metricas para evaluar salidas multilabel."""

import numpy as np


def apply_threshold(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convierte probabilidades a etiquetas binarias usando un umbral fijo."""

    return (probabilities >= threshold).astype(np.float32)


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Hamming loss.

    La idea es simple:
    - comparar cada componente del vector real con la predicha,
    - contar cuantas componentes quedan distintas,
    - y promediar ese error.
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred deben tener la misma forma para calcular Hamming loss."
        )

    mistakes = np.not_equal(y_true, y_pred)
    return float(mistakes.mean())


def exact_match_accuracy(*args, **kwargs):
    """TODO: implementar exact match accuracy si se decide usar esta metrica."""

    raise NotImplementedError("TODO: implementar exact_match_accuracy().")


def f1_multilabel(*args, **kwargs):
    """TODO: implementar F1 multilabel cuando el curso lo requiera."""

    raise NotImplementedError("TODO: implementar f1_multilabel().")
