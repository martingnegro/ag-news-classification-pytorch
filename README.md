# Clasificación de noticias AG News con PyTorch

## Objetivo

Construir un pipeline reproducible de entrenamiento y validación
para clasificar noticias en cuatro categorías mediante una red
neuronal en PyTorch.

## Dataset

Se utiliza AG News, un dataset de noticias en inglés con cuatro
categorías: Internacional, Deportes, Negocios y Ciencia/Tecnología.

Los archivos incluyen las columnas `Class Index`, `Title` y
`Description`:

- `data/ag_news_train.csv`: 120.000 noticias, 30.000 por categoría.
- `data/ag_news_test.csv`: 7.600 noticias, 1.900 por categoría.

## Organización del repositorio

- `data/`: archivos del dataset.
- `notebooks/01_pipeline_base_ag_news.ipynb`: código, resultados y gráficos.
- `requirements.txt`: dependencias.
- `README.md`: documentación del experimento.

## Preparación de los datos

Se combinan el título y la descripción de cada noticia.

El archivo de entrenamiento se divide de forma estratificada,
con semilla 42:

| Conjunto | Noticias | Noticias por categoría |
|---|---:|---:|
| Entrenamiento | 84.000 | 21.000 |
| Validación | 36.000 | 9.000 |

Las etiquetas originales, de 1 a 4, se convierten al rango de 0 a 3.

El archivo `ag_news_test.csv` permanece reservado. No se utiliza
para entrenar, ajustar el vocabulario ni evaluar esta primera etapa.

## Representación del texto

Se utiliza `CountVectorizer` con presencia binaria de palabras,
conversión a minúsculas y un máximo de 5.000 términos.

El vocabulario se aprende únicamente con entrenamiento y se aplica
sin reajustarlo a validación.

Las matrices se conservan en formato disperso y se convierten a
tensores por lotes para controlar el uso de memoria.

## Modelo y configuración

Se implementa un MLP mediante `nn.Module`:

- Entrada: 5.000 características.
- Capa oculta: 32 neuronas con ReLU.
- Salida: cuatro clases.
- Parámetros entrenables: 160.164.

| Parámetro | Valor |
|---|---|
| Función de pérdida | CrossEntropyLoss |
| Optimizador | Adam |
| Learning rate | 0.001 |
| Épocas | 10 |
| Tamaño de lote | 256 |
| Semilla | 42 |

La función de pérdida recibe las salidas del modelo sin aplicar
Softmax previamente.

El dispositivo se selecciona automáticamente entre CUDA, MPS y CPU.
El modelo y los lotes se colocan en el mismo dispositivo.

## Entrenamiento y validación

Cada época recorre todos los lotes de entrenamiento e implementa:

1. Forward pass.
2. Cálculo de pérdida.
3. Limpieza de gradientes con `optimizer.zero_grad()`.
4. Cálculo de gradientes con `loss.backward()`.
5. Actualización de pesos con `optimizer.step()`.

La función `evaluate` calcula pérdida y exactitud sobre validación
con `model.eval()` y `torch.no_grad()`, sin actualizar parámetros.

La pérdida media se pondera por la cantidad de noticias de cada lote,
incluido el último lote, que puede ser más pequeño.

Las métricas de entrenamiento se acumulan mientras el modelo aprende
durante la época. Las de validación se calculan con el modelo al
finalizarla.

Se conserva el modelo al terminar las diez épocas, sin selección
automática ni restauración de una época anterior.

## Resultados del experimento

Ejecución realizada en CPU con PyTorch 2.11.0+cpu.

| Métrica | Entrenamiento | Validación |
|---|---:|---:|
| Pérdida en la primera época | 0.5170 | 0.3068 |
| Pérdida en la última época | 0.1248 | 0.3323 |
| Exactitud en la última época | 95,63 % | 89,82 % |

La pérdida de entrenamiento disminuyó durante las diez épocas.
En validación, descendió inicialmente y luego aumentó de forma
sostenida, mientras la exactitud dejó de mejorar y disminuyó
ligeramente.

Este comportamiento indica sobreajuste en la ejecución observada:
la mejora sobre los datos de entrenamiento no se trasladó a
validación.

El pipeline permite detectar y documentar ese comportamiento.
Los resultados son de validación y no constituyen una evaluación
final sobre el conjunto de prueba reservado.

El notebook incluye las curvas de pérdida y exactitud, utilizando
azul para entrenamiento y naranja para validación.

## Ejecución en Google Colab

1. Abrir Google Colab.
2. Seleccionar la opción para abrir un notebook desde GitHub.
3. Introducir la URL del repositorio:
   https://github.com/martingnegro/ag-news-classification-pytorch
4. Abrir `notebooks/01_pipeline_base_ag_news.ipynb`.
5. Conectar el entorno y ejecutar todas las celdas en orden.


## Dependencias y reproducibilidad

Las versiones de las dependencias se encuentran en `requirements.txt`.
Para instalarlas en un entorno Python compatible:

    python -m pip install -r requirements.txt

Para ejecutar el notebook localmente también se necesita Jupyter
o un editor compatible con archivos `.ipynb`.

Se fijan semillas para Python, NumPy, PyTorch y la mezcla de los
lotes. Pueden existir diferencias entre dispositivos y versiones.

Para repetir el experimento desde cero, ejecutar todas las celdas
en orden. Esto reinicia el modelo, el optimizador y la secuencia
de mezcla de los datos.
