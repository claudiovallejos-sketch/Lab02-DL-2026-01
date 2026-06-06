# Laboratorio 02 — Deep Learning UCN
## Redes Neuronales Poco Profundas para el Estudio del Deterioro Cognitivo

**Asignatura:** Deep Learning  
**Profesor:** Dr. Juan Bekios Calfa  
**Autores:** Claudio Vallejos, José Meléndez  
**Universidad Católica del Norte**

---

## Descripción

Este laboratorio implementa redes neuronales poco profundas en PyTorch para estudiar el deterioro cognitivo mediante seis experimentos independientes, uno por cada variable objetivo (GDS, GDS_R1, GDS_R2, GDS_R3, GDS_R4, GDS_R5), codificadas como vectores multilabel tipo one-hot.

## Estructura del Proyecto

    Lab02-DL-2026-01/
    ├── main.py                  # Punto de entrada principal
    ├── run_all_experiments.py   # Compara los 6 experimentos
    ├── predict_examples.py      # Ejemplos de prediccion con incertidumbre
    ├── dataset/
    │   └── deterioro_cognitivo.sav
    ├── src/
    │   ├── config.py            # Configuracion y valores por defecto
    │   ├── data_loader.py       # Carga del dataset
    │   ├── preprocessing.py     # Transformaciones y validacion cruzada
    │   ├── models.py            # Red neuronal en PyTorch
    │   ├── evaluation.py        # Metricas multilabel
    │   └── uncertainty.py       # Monte Carlo Dropout
    ├── environment.yml
    └── README.md

## Instalación

    conda env create -f environment.yml
    conda activate lab_pytorch

## Uso

### Ejecutar un experimento individual

    python main.py --data-path dataset/deterioro_cognitivo.sav --target-name GDS_R2 --epochs 50

### Comparar todos los experimentos

    python run_all_experiments.py

### Ver ejemplos de predicción con incertidumbre

    python predict_examples.py

## Características Implementadas

- Codificacion multilabel tipo one-hot
- Red neuronal poco profunda con PyTorch (ShallowMultiLabelNet)
- Validacion cruzada anidada (5 folds externos, 3 internos)
- Grid search de hiperparametros (hidden_dim, dropout, learning_rate)
- Metricas multilabel: Hamming Loss, Exact Match, F1 Macro, F1 Micro, Precision Micro, Recall Micro
- Monte Carlo Dropout para estimacion de incertidumbre

## Resultados

| Target | Clases | Hamming Mean | Hamming Std |
|--------|--------|--------------|-------------|
| GDS_R1 | 3      | 0.0697       | 0.0162      |
| GDS_R2 | 3      | 0.1853       | 0.0184      |
| GDS_R3 | 2      | 0.0929       | 0.0248      |
| GDS_R4 | 3      | 0.1141       | 0.0068      |
| GDS_R5 | 3      | 0.1510       | 0.0087      |

## Hiperparámetros Explorados

| Hiperparametro | Valores       |
|----------------|---------------|
| Hidden Dim     | 16, 32, 64    |
| Dropout        | 0.2, 0.3, 0.5 |
| Learning Rate  | 1e-2, 1e-3    |
