"""Funciones para preparar el target multilabel del experimento activo."""

import numpy as np
import pandas as pd

from .config import TARGET_COLUMNS
from .data_loader import build_input_matrix


def encode_target_as_one_hot(
    dataframe: pd.DataFrame, target_name: str
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Toma una sola columna objetivo y la convierte a formato one-hot.

    Ejemplo:
    Si la clase original es 2 y el experimento tiene tres clases,
    entonces el vector queda como [0, 1, 0].
    """

    if target_name not in TARGET_COLUMNS:
        raise ValueError(
            f"Target invalido: {target_name}. Debe ser uno de {TARGET_COLUMNS}."
        )

    if target_name not in dataframe.columns:
        raise ValueError(f"La columna objetivo {target_name} no existe en el dataset.")

    y_raw = dataframe[target_name].astype(int)
    classes = sorted(y_raw.unique().tolist())
    class_to_idx = {class_value: idx for idx, class_value in enumerate(classes)}

    y_idx = y_raw.map(class_to_idx).to_numpy()
    y_encoded = np.zeros((len(y_idx), len(classes)), dtype=np.float32)
    y_encoded[np.arange(len(y_idx)), y_idx] = 1.0

    return y_encoded, classes, class_to_idx


def prepare_experiment_data(
    dataframe: pd.DataFrame, target_name: str
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int]]:
    """Prepara X e Y para un experimento puntual."""

    X = build_input_matrix(dataframe)
    Y, classes, class_to_idx = encode_target_as_one_hot(dataframe, target_name)
    return X, Y, classes, class_to_idx


def split_for_validation(*args, **kwargs):
    """
    TODO:
    Implementar una separacion honesta entre entrenamiento, validacion interna
    y prueba externa.

    La presentacion enfatiza que esta parte es tan importante como el modelo.
    """

    raise NotImplementedError(
        "TODO: implementar split_for_validation() con una estrategia experimental."
    )
