"""Punto de entrada simple del proyecto."""

import argparse
from pathlib import Path

from .config import DEFAULT_DROPOUT, DEFAULT_HIDDEN_DIM, DEFAULT_TARGET
from .data_loader import CognitiveMultiLabelDataset, load_dataframe
from .models import ShallowMultiLabelNet
from .preprocessing import prepare_experiment_data


def build_project_objects(
    data_path: str | Path,
    target_name: str = DEFAULT_TARGET,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout: float = DEFAULT_DROPOUT,
) -> dict:
    """Construye los objetos base del experimento sin entrenar el modelo."""

    dataframe = load_dataframe(data_path)
    X, Y, classes, class_to_idx = prepare_experiment_data(dataframe, target_name)

    dataset = CognitiveMultiLabelDataset(X, Y)
    model = ShallowMultiLabelNet(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        output_dim=Y.shape[1],
    )

    return {
        "dataframe": dataframe,
        "dataset": dataset,
        "model": model,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "X_shape": X.shape,
        "Y_shape": Y.shape,
    }


def train_one_experiment(*args, **kwargs):
    """
    TODO:
    Implementar el entrenamiento con:
    - DataLoader
    - BCEWithLogitsLoss
    - optimizador
    - validacion interna
    - validacion externa
    """

    raise NotImplementedError(
        "TODO: implementar train_one_experiment() segun la metodologia del laboratorio."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Define los argumentos de linea de comandos."""

    parser = argparse.ArgumentParser(
        description="Proyecto base del laboratorio de redes neuronales poco profundas."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Ruta al archivo de datos. Debe estar dentro de dataset/.",
    )
    parser.add_argument(
        "--target-name",
        default=DEFAULT_TARGET,
        help=f"Experimento activo. Por defecto usa {DEFAULT_TARGET}.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help="Numero de neuronas de la capa oculta.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Probabilidad de dropout.",
    )
    return parser


def main() -> None:
    """Ejecuta solo la parte ya implementada del proyecto."""

    parser = build_argument_parser()
    args = parser.parse_args()

    artifacts = build_project_objects(
        data_path=args.data_path,
        target_name=args.target_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    print("Proyecto base cargado correctamente.")
    print(f"Experimento activo: {args.target_name}")
    print(f"Forma de X: {artifacts['X_shape']}")
    print(f"Forma de Y: {artifacts['Y_shape']}")
    print(f"Clases encontradas: {artifacts['classes']}")
    print("Modelo creado, pero el entrenamiento aun esta pendiente.")
    print("TODO: completar train_one_experiment() y uncertainty.py.")


if __name__ == "__main__":
    main()
