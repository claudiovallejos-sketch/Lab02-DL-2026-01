"""Punto de entrada minimo del proyecto."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INNER_FOLDS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHT_DECAY,
)
from data_loader import CognitiveMultiLabelDataset, load_dataframe  # noqa: E402
from evaluation import apply_threshold, hamming_loss  # noqa: E402
from models import ShallowMultiLabelNet  # noqa: E402
from preprocessing import prepare_experiment_data, split_for_validation  # noqa: E402


def build_project_objects(
    data_path: str | Path,
    target_name: str = DEFAULT_TARGET,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout: float = DEFAULT_DROPOUT,
) -> dict:
    """Construye los objetos base del experimento."""

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
        "X": X,
        "Y": Y,
        "X_shape": X.shape,
        "Y_shape": Y.shape,
    }


def set_seed(seed: int) -> None:
    """Fija semillas para que la ejecucion sea reproducible."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resuelve el dispositivo solicitado por linea de comandos."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se solicito CUDA, pero no hay GPU disponible.")

    return device


def build_data_loader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Construye un DataLoader simple para entrenamiento o evaluacion."""

    dataset = CognitiveMultiLabelDataset(X, Y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Entrena una epoca con BCEWithLogitsLoss."""

    model.train()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


def evaluate_with_hamming(
    model: nn.Module,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
) -> float:
    """Evalua el modelo con Hamming Loss."""

    model.eval()
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            all_probabilities.append(probabilities)
            all_targets.append(targets.numpy())

    probabilities_np = np.vstack(all_probabilities)
    targets_np = np.vstack(all_targets)
    predictions_np = apply_threshold(probabilities_np, threshold=threshold)
    return hamming_loss(targets_np, predictions_np)


def run_training_cycle(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    threshold: float,
    seed: int,
    device: torch.device,
) -> dict:
    """Entrena y evalua una configuracion puntual del modelo."""

    set_seed(seed)

    train_loader = build_data_loader(
        X_train, Y_train, batch_size=batch_size, shuffle=True, seed=seed
    )
    eval_loader = build_data_loader(
        X_eval, Y_eval, batch_size=batch_size, shuffle=False, seed=seed
    )

    model = ShallowMultiLabelNet(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        output_dim=Y_train.shape[1],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []
    for _ in range(epochs):
        epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        history.append(epoch_loss)

    eval_hamming = evaluate_with_hamming(
        model,
        eval_loader,
        threshold=threshold,
        device=device,
    )

    return {
        "model": model,
        "train_losses": history,
        "hamming_loss": eval_hamming,
        "final_train_loss": history[-1],
    }


def train_one_experiment(
    data_path: str | Path,
    target_name: str = DEFAULT_TARGET,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    threshold: float = DEFAULT_THRESHOLD,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    seed: int = DEFAULT_RANDOM_SEED,
    device_name: str = "cpu",
) -> dict:
    """
    Ejecuta la version minima del flujo de validacion del laboratorio.

    La presentacion propone validacion cruzada anidada. En esta plantilla
    se deja una sola configuracion fija para que los alumnos entiendan
    primero el flujo completo antes de implementar la busqueda de
    hiperparametros.
    """

    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor o igual que 1.")

    if epochs < 1:
        raise ValueError("epochs debe ser mayor o igual que 1.")

    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds e inner_folds deben ser al menos 2.")

    artifacts = build_project_objects(
        data_path=data_path,
        target_name=target_name,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    X = artifacts["X"]
    Y = artifacts["Y"]
    device = resolve_device(device_name)

    outer_splits = split_for_validation(
        Y, n_splits=outer_folds, random_state=seed
    )
    outer_results = []

    for outer_fold_index, (outer_train_idx, outer_test_idx) in enumerate(
        outer_splits, start=1
    ):
        X_outer_train = X[outer_train_idx]
        Y_outer_train = Y[outer_train_idx]
        X_outer_test = X[outer_test_idx]
        Y_outer_test = Y[outer_test_idx]

        inner_splits = split_for_validation(
            Y_outer_train,
            n_splits=inner_folds,
            random_state=seed + outer_fold_index,
        )

        inner_hamming_scores = []
        # La validacion interna ya esta lista para que luego los alumnos
        # agreguen comparacion de hiperparametros sin rehacer el flujo.
        for inner_fold_index, (inner_train_idx, inner_val_idx) in enumerate(
            inner_splits, start=1
        ):
            inner_result = run_training_cycle(
                X_train=X_outer_train[inner_train_idx],
                Y_train=Y_outer_train[inner_train_idx],
                X_eval=X_outer_train[inner_val_idx],
                Y_eval=Y_outer_train[inner_val_idx],
                hidden_dim=hidden_dim,
                dropout=dropout,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                threshold=threshold,
                seed=seed + outer_fold_index * 100 + inner_fold_index,
                device=device,
            )
            inner_hamming_scores.append(inner_result["hamming_loss"])

        # En la plantilla minima no se elige entre varias configuraciones;
        # se reentrena la misma red con todo el fold externo de entrenamiento.
        final_result = run_training_cycle(
            X_train=X_outer_train,
            Y_train=Y_outer_train,
            X_eval=X_outer_test,
            Y_eval=Y_outer_test,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            threshold=threshold,
            seed=seed + outer_fold_index * 1000,
            device=device,
        )

        outer_results.append(
            {
                "outer_fold": outer_fold_index,
                "inner_hamming_mean": float(np.mean(inner_hamming_scores)),
                "inner_hamming_std": float(np.std(inner_hamming_scores)),
                "outer_hamming_loss": final_result["hamming_loss"],
                "final_train_loss": final_result["final_train_loss"],
            }
        )

    outer_hamming_losses = np.asarray(
        [fold_result["outer_hamming_loss"] for fold_result in outer_results],
        dtype=np.float32,
    )

    return {
        "target_name": target_name,
        "classes": artifacts["classes"],
        "X_shape": artifacts["X_shape"],
        "Y_shape": artifacts["Y_shape"],
        "device": str(device),
        "config": {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "threshold": threshold,
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "seed": seed,
        },
        "outer_folds": outer_results,
        "summary": {
            "mean_hamming_loss": float(outer_hamming_losses.mean()),
            "std_hamming_loss": float(outer_hamming_losses.std()),
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Define los argumentos de linea de comandos para el experimento base."""

    parser = argparse.ArgumentParser(
        description=(
            "Version minima del laboratorio: red poco profunda, "
            "validacion anidada y Hamming Loss."
        )
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate del optimizador Adam.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay del optimizador Adam.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Tamano de batch para DataLoader.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Numero de epocas de entrenamiento por fold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Umbral para convertir probabilidades en etiquetas binarias.",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=DEFAULT_OUTER_FOLDS,
        help="Cantidad de folds externos para la evaluacion final.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=DEFAULT_INNER_FOLDS,
        help="Cantidad de folds internos para la validacion.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Dispositivo para ejecutar el experimento.",
    )
    return parser


def main() -> None:
    """Ejecuta el flujo minimo del laboratorio."""

    parser = build_argument_parser()
    args = parser.parse_args()

    results = train_one_experiment(
        data_path=args.data_path,
        target_name=args.target_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        threshold=args.threshold,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        device_name=args.device,
    )

    print("Experimento ejecutado correctamente.")
    print(f"Experimento activo: {results['target_name']}")
    print(f"Dispositivo usado: {results['device']}")
    print(f"Forma de X: {results['X_shape']}")
    print(f"Forma de Y: {results['Y_shape']}")
    print(f"Clases encontradas: {results['classes']}")
    print(
        "Hamming Loss externo promedio: "
        f"{results['summary']['mean_hamming_loss']:.4f} "
        f"+/- {results['summary']['std_hamming_loss']:.4f}"
    )

    for fold_result in results["outer_folds"]:
        print(
            f"Fold externo {fold_result['outer_fold']}: "
            f"Hamming interno = {fold_result['inner_hamming_mean']:.4f} "
            f"+/- {fold_result['inner_hamming_std']:.4f}, "
            f"Hamming externo = {fold_result['outer_hamming_loss']:.4f}, "
            f"loss final de entrenamiento = {fold_result['final_train_loss']:.4f}"
        )

    print("TODO: agregar mas metricas, busqueda de hiperparametros e incertidumbre.")


if __name__ == "__main__":
    main()
