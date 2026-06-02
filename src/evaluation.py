"""Metricas para evaluar salidas multilabel."""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def apply_threshold(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convierte probabilidades a etiquetas binarias usando un umbral fijo."""
    return (probabilities >= threshold).astype(np.float32)


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula Hamming loss."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred deben tener la misma forma.")
    return float(np.not_equal(y_true, y_pred).mean())


def exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact match: solo cuenta como correcto si TODAS las etiquetas coinciden."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.all(y_true == y_pred, axis=1).mean())


def f1_multilabel(y_true: np.ndarray, y_pred: np.ndarray,
                  average: str = "macro") -> float:
    """F1 score para clasificacion multilabel."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def precision_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision Micro para clasificacion multilabel."""
    return float(precision_score(y_true, y_pred, average="micro", zero_division=0))


def recall_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall Micro para clasificacion multilabel."""
    return float(recall_score(y_true, y_pred, average="micro", zero_division=0))
