# Lab02-DL-2026-01

Proyecto base para el Laboratorio 02 de Deep Learning.

La idea del laboratorio, siguiendo la presentacion, es tratar el problema como
seis experimentos independientes:

- `GDS`
- `GDS_R1`
- `GDS_R2`
- `GDS_R3`
- `GDS_R4`
- `GDS_R5`

Cada experimento toma una sola columna objetivo y la transforma a formato
one-hot. Este repositorio deja una plantilla simple y modular para que los
estudiantes completen el laboratorio paso a paso.

## Que esta implementado

- Carga basica de datos desde `csv` o `sav`
- Seleccion de columnas de entrada
- Codificacion one-hot del target activo
- `Dataset` de PyTorch
- Red neuronal poco profunda
- `hamming_loss` para evaluar predicciones multilabel

## Que queda como TODO

- Separacion honesta en entrenamiento, validacion interna y prueba externa
- Bucle de entrenamiento con `BCEWithLogitsLoss`
- Comparacion entre los seis experimentos
- Metricas adicionales
- Algoritmo de incertidumbre con Monte Carlo Dropout

## Estructura

```text
Lab02-DL-2026-01/
|-- dataset/
|   `-- README.md
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- evaluation.py
|   |-- main.py
|   |-- models.py
|   |-- preprocessing.py
|   `-- uncertainty.py
|-- .gitignore
|-- environment.yml
`-- README.md
```

## Uso esperado

1. Copiar el dataset dentro de `dataset/`.
2. Activar el entorno del proyecto.
3. Ejecutar el flujo base.

```bash
python -m src.main --data-path dataset/archivo.csv --target-name GDS_R2
```

Ese comando:

- carga el dataset,
- prepara `X` y `Y`,
- crea el `Dataset` de PyTorch,
- construye la red poco profunda,
- y deja indicado en pantalla lo que falta por implementar.

## Plan sugerido para alumnos

1. Comprender el problema y revisar las columnas del dataset.
2. Elegir un experimento objetivo, por ejemplo `GDS_R2`.
3. Verificar la codificacion one-hot del target.
4. Completar el entrenamiento del modelo.
5. Agregar una validacion experimental mas honesta.
6. Comparar resultados entre experimentos.
7. Implementar incertidumbre al final del laboratorio.
