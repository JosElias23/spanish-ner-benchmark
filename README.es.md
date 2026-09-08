# Reconocimiento de entidades nombradas en español: de tablas de búsqueda a transformers

[![CI](https://github.com/JosElias23/spanish-ner-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/JosElias23/spanish-ner-benchmark/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-60%20passing-brightgreen)](https://github.com/JosElias23/spanish-ner-benchmark/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[English](README.md) · **Español**

NER de punta a punta para español sobre el benchmark CoNLL-2002. Cinco modelos,
un solo protocolo de evaluación, una única lectura del conjunto de test reservado
e intervalos de confianza en cada comparación.

Pruébalo: `python app/app.py` para una demo interactiva en Gradio, o `docker compose
-f serve/docker-compose.yml up --build` para el servicio de producción con sus
números de latencia y costo. Ambos se describen más abajo.

---

## Resultados

F1 a nivel de entidad sobre el split de test oficial (`esp.testb`), leído
exactamente una vez al final del proyecto.

| Modelo | Parámetros | F1 test | F1 test (dedup.) | Precisión | Recall | Tiempo de entrenamiento |
|---|---:|---:|---:|---:|---:|---:|
| Gazetteer (longest match) | — | 0,3595 | 0,3364 | 0,2634 | 0,5662 | 0,05 s |
| CRF (features diseñadas a mano) | — | 0,7924 | 0,7769 | 0,7966 | 0,7881 | 51 s |
| **BETO** (`bert-base-spanish-wwm-cased`) | 109 M | **0,8705** | 0,8642 | 0,8626 | 0,8786 | 105 s |
| **mBERT** (`bert-base-multilingual-cased`) | 177 M | **0,8720** | 0,8659 | 0,8634 | 0,8809 | 110 s |
| XLM-R (`xlm-roberta-base`) | 277 M | 0,8622 | 0,8530 | 0,8568 | 0,8677 | 452 s |

Los tiempos son sobre una sola RTX 5060 Ti (8 GB), cuatro épocas cada uno.

![Comparación de modelos](reports/figures/model_comparison.png)

La columna *deduplicado* se explica en [Un problema con el
benchmark](#un-problema-con-el-benchmark).

Todos los números los produce `scripts/evaluate_test.py` y quedan guardados
literalmente en [`reports/metrics_test.json`](reports/metrics_test.json). Nada de
este README está escrito a mano.

---

## Tres hallazgos

### 1. El mejor modelo en el conjunto de desarrollo fue el peor en test

| Modelo | F1 dev | F1 test | Cambio de puesto |
|---|---:|---:|:--|
| XLM-R | **0,8765** (1.º) | 0,8622 (3.º) | ▼ 2 |
| mBERT | 0,8665 (2.º) | **0,8720** (1.º) | ▲ 1 |
| BETO | 0,8640 (3.º) | 0,8705 (2.º) | ▲ 1 |

Elegir un modelo solo por su desempeño en el conjunto de desarrollo, que es lo
que hace la mayoría de los tutoriales, habría dejado en producción el peor de los
tres transformers. Esto es sobreajuste en la selección de modelo sobre un
conjunto de desarrollo de 1.915 oraciones, observado directamente en vez de
descrito en abstracto.

XLM-R es además, por lejos, el modelo más caro de la comparación: 277 M de
parámetros y 452 s de entrenamiento frente a los 109 M y 105 s de BETO, un costo
4,3× mayor para el peor puntaje en test. Más grande y más multilingüe fue, aquí,
estrictamente peor en todos los ejes que importan.

### 2. Un encoder específico para español no da ninguna ventaja medible aquí

Bootstrap pareado, 10.000 remuestreos, contra el modelo mejor rankeado:

| Comparación | ΔF1 | IC 95 % | p | Significativo |
|---|---:|:--|---:|:--:|
| mBERT vs BETO | +0,0015 | [−0,0086, +0,0113] | 0,75 | **no** |
| mBERT vs XLM-R | +0,0098 | [+0,0003, +0,0195] | 0,04 | sí |
| mBERT vs CRF | +0,0797 | [+0,0624, +0,0978] | <0,001 | sí |
| mBERT vs Gazetteer | +0,5125 | [+0,4931, +0,5316] | <0,001 | sí |

BETO y mBERT son estadísticamente indistinguibles. El proyecto partió para poner
a prueba la hipótesis de que un modelo específico para español le gana a uno
multilingüe en NER en español, y **en este benchmark esa hipótesis no se
sostiene.**

Reportar `mBERT 0,8720 > BETO 0,8705` como resultado habría sido una afirmación
sobre ruido de muestreo. Una diferencia de un punto de F1 sobre 1.517 oraciones
normalmente lo es.

### 3. La detección está resuelta; la clasificación no

![Desglose de errores](reports/figures/error_breakdown.png)

De los 479 errores de BETO en el conjunto de test:

| Tipo de error | Conteo | Proporción | Qué significa |
|---|---:|---:|---|
| Tipo | 230 | 48,0 % | Span correcto, clase equivocada |
| Límite | 186 | 38,8 % | Clase correcta, span equivocado |
| Espurio | 47 | 9,8 % | Entidad predicha donde no hay ninguna |
| **Omitido** | **16** | **3,3 %** | Entidad del gold que no se encuentra |

El modelo casi siempre encuentra la entidad: solo el 3,3 % de los errores son
omisiones lisas y llanas. Casi nueve de cada diez errores son sobre *etiquetar*
algo que ya localizó. El esfuerzo puesto en recall estaría en gran medida
desperdiciado; el retorno está en desambiguar tipos, sobre todo ORG vs LOC, y en
los límites de los spans.

![F1 por tipo de entidad](reports/figures/f1_by_entity_type.png)

`MISC` es la clase más débil para todos los modelos (F1 de BETO = 0,671 frente a
0,958 en `PER`). Es lo esperable: `MISC` está definida en negativo en las guías
de anotación. Es todo lo que es una entidad pero no una persona, una organización
ni un lugar. Por eso no tiene una forma superficial consistente que aprender.

El registro completo de decisiones, incluidas las que cambiaron estos números y
las que resultaron no cambiarlos, está en [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## La tarea

Dado un texto en español, etiquetar cada mención de una entidad de tipo
**persona**, **organización**, **lugar** o **misceláneo** (nacionalidades,
eventos, obras de arte).

```
El   Banco Central de Chile   anuncio hoy en   Santiago
     └────── ORG ─────────┘                    └─ LOC ─┘
```

NER es la capa de extracción que hay debajo de la búsqueda documental, el
screening de cumplimiento, KYC y la analítica basada en entidades. Es también el
banco de pruebas estándar para etiquetado de secuencias, lo que lo convierte en
un buen benchmark para comparar familias de modelos bajo un protocolo fijo.

## Los datos

[CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/) en español, a partir
de textos del cable noticioso EFE (mayo de 2000), anotados por la Universidad de
Amberes.

| Split | Archivo | Oraciones | Tokens | Entidades |
|---|---|---:|---:|---:|
| Train | `esp.train` | 8.323 | 264.715 | 18.798 |
| Development | `esp.testa` | 1.915 | 52.923 | 4.352 |
| Test | `esp.testb` | 1.517 | 51.533 | 3.559 |

Las entidades están etiquetadas en IOB2 sobre cuatro tipos: LOC (4.914), ORG
(7.390), PER (4.321), MISC (2.173) en el split de entrenamiento.

El corpus se parsea desde los archivos de columnas en crudo con
`src/spanish_ner/data.py` en lugar de usar un script de carga alojado, y cada
archivo se verifica por SHA-256 al cargarlo. Esto importa en la práctica:
`datasets>=3` eliminó el soporte para datasets basados en script, lo que rompió
por completo el camino convencional `load_dataset("conll2002", "es")`.

> **Un detalle de parseo que costó F1 real.** Estos archivos son UTF-8, pese a que
> este corpus casi siempre se lee como latin-1. Decodificarlos como latin-1
> corrompe en silencio unos 27.000 tokens acentuados, `información` queda como
> `informaciÃ³n`, y nada levanta una excepción. El entrenamiento corre, la pérdida
> baja y el puntaje queda calladamente peor.
> `tests/test_data.py::test_encoding_is_utf8_not_mojibake` es el cable trampa.

### Un problema con el benchmark

Los splits oficiales **no son disjuntos**. 309 de las 1.517 oraciones de test
(20,4 %) aparecen textualmente en los datos de entrenamiento:

| | Conteo |
|---|---:|
| Oraciones de test duplicadas en train | 309 (20,4 %) |
| ...que son el único token `-` | 162 |
| ...que son datelines de agencia (`Madrid , 23 may ( EFE ) .`) | ~52 |
| ...con 10 o más tokens (contenido real) | 86 |
| **Entidades de test dentro de oraciones duplicadas** | **336 / 3.559 (9,4 %)** |

La mayor parte del solapamiento es relleno de cable noticioso, pero el 9,4 % de
las entidades de test está en oraciones que el modelo ya vio. Sacarlas del
benchmark oficial rompería la comparabilidad con los resultados publicados, así
que cada tabla reporta las dos cifras:

- **Completo**: las 1.517 oraciones oficiales, comparable con la literatura.
- **Deduplicado**: las 1.208 oraciones ausentes del entrenamiento, una estimación
  más estricta de la generalización a texto genuinamente nuevo.

El ranking no cambia, pero el *costo* de la deduplicación no es uniforme, y el
patrón dice algo:

| Modelo | Completo | Deduplicado | Caída |
|---|---:|---:|---:|
| Gazetteer | 0,3595 | 0,3364 | **−2,31 pts** |
| CRF | 0,7924 | 0,7769 | −1,55 pts |
| XLM-R | 0,8622 | 0,8530 | −0,92 pts |
| BETO | 0,8705 | 0,8642 | −0,63 pts |
| mBERT | 0,8720 | 0,8659 | −0,61 pts |

**Mientras más se apoya un modelo en la memorización, más pierde cuando se quitan
las oraciones memorizadas.** El gazetteer, que no es más que memorización, cede
casi cuatro veces más que los transformers. Esta es una medición directa y
cuantitativa de la generalización, obtenida gratis a partir de un defecto del
benchmark.

Estos conteos están fijados en `tests/test_data.py::TestSplitOverlap`, así que si
los datos crudos alguna vez cambian falla la suite de tests en lugar de derivar
las métricas.

---

## Enfoque

Cuatro familias de modelado, en orden creciente de capacidad. Cada una existe
para hacer interpretable el puntaje de la siguiente.

**1. Gazetteer.** Memorizar todas las formas superficiales de entidad del
conjunto de entrenamiento; longest match en inferencia; resolver la ambigüedad de
tipo por voto de mayoría. Este es el piso, y cuantifica cuánto de la tarea es
pura memorización (F1 = 0,36. Cerca del 41 % del puntaje de BETO sin aprender
nada).

**2. CRF.** Un campo aleatorio condicional de cadena lineal sobre features
diseñadas a mano: mayúsculas, prefijos y sufijos de hasta tres caracteres,
patrones de dígitos y guiones, la etiqueta POS PAROLE completa y truncada a dos
caracteres, y todo lo anterior para una ventana de ±2 tokens. Las features de
sufijo cargan más señal en español que en inglés por su morfología flexiva más
rica. Un CRF puntúa la secuencia de etiquetas completa de forma conjunta, así que
aprende restricciones estructurales como *I-PER nunca sigue a B-ORG* directamente
desde los pesos de transición. Esto era el estado del arte antes de los
embeddings contextuales, y es la vara real que un transformer tiene que pasar.

**3–5. Transformers.** BETO, mBERT y XLM-R con fine-tuning para clasificación de
tokens, todos a través del mismo script para que ningún resultado pueda diferir
por un cambio accidental de procedimiento.

### Alineación de etiquetas a sub-tokens

El problema técnico central. El corpus está anotado por palabra; BERT tokeniza en
sub-palabras:

```
words       ["Telefónica",  "invirtió"          ]
labels      ["B-ORG",       "O"                 ]
sub-tokens  ["Telef", "##ónica", "invir", "##tió"]
```

Las etiquetas hay que re-proyectarlas sobre los sub-tokens. Etiquetamos el
**primer** sub-token de cada palabra y asignamos `-100` al resto; la
cross-entropy de PyTorch ignora `-100`, así que los sub-tokens de continuación no
aportan pérdida ni gradiente.

La alternativa, repetir la etiqueta en cada sub-token, sobrepondera las palabras
largas en la pérdida y vuelve ambigua la decodificación cuando los sub-tokens de
una misma palabra no coinciden. Además obliga a convertir `B-` en `I-` en las
continuaciones, o la secuencia decodificada termina con dos etiquetas `B-ORG`
adyacentes y parte una entidad en dos. Ambas estrategias están implementadas;
`--label-all-subtokens` corre la ablación.

Las etiquetas mal alineadas son el fallo silencioso clásico del fine-tuning en
NER: el modelo entrena, la pérdida baja y el target estuvo malo todo el tiempo.
Diez tests en `tests/test_modeling.py` fijan el contrato de alineación, incluido
uno que verifica que la palabra de prueba efectivamente se parte en sub-tokens.
Si no, los otros tests pasarían sin probar nada.

### Protocolo de evaluación

- **Puntaje a nivel de entidad** con `seqeval`, igual que el script oficial de
  CoNLL. Una predicción cuenta solo si el tipo *y* ambos límites son exactos. La
  accuracy a nivel de token no dice nada acá: ~88 % de los tokens están fuera de
  toda entidad, así que un modelo que prediga `O` en todas partes saca 0,88 de
  accuracy sin encontrar nada.
- **El split de test lo carga exactamente un script.** `scripts/evaluate_test.py`
  es el único archivo del repositorio que lee `esp.testb`. La selección de
  baselines, los hiperparámetros, el early stopping y el análisis de errores
  corren todos sobre dev.
- **Se conserva la mejor época por F1 en dev**, no la última, para que un modelo
  que llega a su máximo en la época 3 y sobreajusta en la 4 no quede reportado en
  su peor momento.
- **Los puntajes reportados se recalculan a través de `predict_sentences`**, la
  misma ruta de inferencia que usa la demo, en vez de confiar en el loop interno
  del `Trainer`. Si alguna vez difieren, el modelo desplegado no es el que se
  midió.
- **Bootstrap pareado** sobre 10.000 remuestreos para cada comparación. Los dos
  modelos se puntúan sobre las mismas oraciones remuestreadas, controlando por el
  hecho de que algunas oraciones son simplemente más difíciles. La
  implementación remuestrea conteos por oración de TP/predichas/gold
  precalculados en lugar de volver a correr `seqeval`, lo que convierte horas en
  milisegundos.
- **Seed fija (42)** en `random`, `numpy` y `torch`.

---

## Reproducir estos números

Todos los comandos de abajo se ejecutaron para producir las tablas de arriba.
Tiempo total de reloj en una RTX 5060 Ti (8 GB): **unos 15 minutos** para los
cinco modelos más la evaluación.

```bash
git clone https://github.com/JosElias23/spanish-ner-benchmark.git
cd spanish-ner-benchmark
```

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,app]"
```

O saltar directo a `make all`, que corre el pipeline completo de punta a punta en
el orden correcto.

```bash
python scripts/download_data.py
```

Descarga los tres archivos crudos del corpus y verifica sus checksums SHA-256.

```bash
python -m pytest
```

60 tests: integridad del corpus, encoding, solapamiento de splits, decodificación
de entidades, alineación de etiquetas a sub-tokens y el contrato del servicio.
Corre esto antes de confiar en cualquier número de más abajo. Los tests de la API
se saltan solos cuando todavía no se ha entrenado ningún checkpoint, así que un
clon nuevo corre en verde.

```bash
python scripts/train_baselines.py
```

Gazetteer y CRF, evaluados en dev. Cerca de un minuto.

```bash
python scripts/train_transformer.py
python scripts/train_transformer.py --model bert-base-multilingual-cased --tag mbert
python scripts/train_transformer.py --model xlm-roberta-base --tag xlmr
```

BETO y mBERT toman menos de dos minutos cada uno en una GPU de consumo moderna;
XLM-R toma unos 7,5 minutos. En CPU funciona, pero es bastante más lento.

```bash
python scripts/evaluate_test.py \
    --models models/bert-base-spanish-wwm-cased/best models/mbert/best models/xlmr/best
```

La única lectura del conjunto de test reservado. Escribe
`reports/metrics_test.json`.

```bash
python scripts/error_analysis.py
```

Regenera todas las figuras de este README más `reports/error_analysis.json`.

```bash
python app/app.py
```

Sirve la demo de Gradio en `http://127.0.0.1:7860`.

---

## Serving: latencia, throughput y costo

Una demo prueba que el modelo corre. Estos son los números que necesita quien
opera el sistema antes de poner algo por delante.

Medido en una sola RTX 5060 Ti, 40 iteraciones por tamaño de batch después de
cinco pasadas de calentamiento descartadas, sobre documentos reales del conjunto
de desarrollo con un promedio de 24,2 tokens. Producido por `serve/benchmark.py`
y guardado en [`reports/metrics_serving.json`](reports/metrics_serving.json).

| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Docs/s | USD por 1M de docs |
|---:|---:|---:|---:|---:|---:|
| 1 | 13,2 | 14,0 | 15,5 | 76 | 1,94 |
| 4 | 14,6 | 15,2 | 17,9 | 274 | 0,54 |
| 8 | 22,8 | 23,4 | 23,7 | 351 | 0,42 |
| 16 | 40,8 | 42,0 | 42,3 | 392 | 0,38 |
| 32 | 68,9 | 69,8 | 70,0 | 464 | 0,32 |
| **64** | 126,6 | 128,8 | 130,0 | **506** | **0,29** |

**Hacer batching de 64 documentos entrega 6,7× el throughput de procesarlos de a
uno y recorta el costo por millón de documentos en un 85 %.** El costo asume
utilización sostenida a USD 0,53/hora para una GPU de inferencia de gama de
entrada, y excluye red, almacenamiento y orquestación.

La forma de esa tabla es el punto. Pasar de batch 1 a batch 4 cuesta 1,4 ms de
latencia y cuadruplica el throughput, porque un batch de cuatro apenas llena la
GPU. Pasar de 32 a 64 duplica la latencia por un 9 % más de throughput. De ahí en
adelante el dispositivo está saturado y el batching solo compra demora de cola.
Un endpoint interactivo debería ubicarse a la izquierda de esta tabla y un
pipeline masivo a la derecha, y ninguno de los dos números por sí solo describe
el sistema.

### Cómo correrlo

```bash
pip install -e ".[serve]"
python serve/benchmark.py                    # reproduce la tabla de arriba
uvicorn serve.api:app --port 8000            # servir localmente
docker compose -f serve/docker-compose.yml up --build
```

El servicio expone `POST /extract`, sondas `/health` y `/ready` separadas, y
`/metrics` en formato de texto Prometheus con p50/p95/p99 móviles sobre las
últimas 10.000 solicitudes. Liveness y readiness son deliberadamente distintas:
los pesos tardan segundos en cargar, y un orquestador que enrute solo por
liveness mandaría tráfico a un pod que no puede responder.

```bash
curl -s localhost:8000/extract -H 'content-type: application/json' \
  -d '{"texts":["El Banco Central de Chile anuncio hoy en Santiago."]}'
```

El contenedor es multi-stage y corre como usuario no root sobre torch CPU-only,
lo que mantiene la imagen lo bastante chica para un tier gratuito y aun así pasa
los 100 documentos por segundo con batch 64.

La inferencia en el servicio pasa por `predict_sentences`, la misma función que
produjo cada métrica de este README, y un test verifica que la extracción con y
sin batch devuelva entidades idénticas. Poner padding a una secuencia corta al
lado de una larga es exactamente donde se esconde un bug de máscara de atención,
y degrada la salida en vez de levantar un error.

---

## Desplegar la demo

El checkpoint entrenado no está commiteado. Pesa 440 MB y es regenerable en menos
de dos minutos con `scripts/train_transformer.py`. Publicar la demo implica, por
lo tanto, publicar primero los pesos.

```bash
huggingface-cli login
huggingface-cli upload JosElias23/spanish-ner-beto \
    models/bert-base-spanish-wwm-cased/best
```

Después crear un Space en
[huggingface.co/new-space](https://huggingface.co/new-space) con SDK **Gradio** y
hardware **CPU basic (free)**, y darle esta estructura:

```
app.py              <- app/app.py, sin cambios
requirements.txt    <- app/requirements.txt, sin cambios
spanish_ner/        <- el paquete src/spanish_ner completo
```

Copiar solo los dos archivos no alcanza. La app importa `predict_sentences` e
`iter_entities` desde el paquete a propósito, para que la demo corra el mismo
código de inferencia que produjo las métricas reportadas, y el paquete tiene que
viajar con ella. En la raíz del Space, `spanish_ner/` es importable directamente;
la línea de `sys.path` en `app.py` no resuelve a nada ahí y queda simplemente
inerte.

Por último, fijar la variable del Space `NER_MODEL_PATH` en
`JosElias23/spanish-ner-beto`.

Esa variable es lo único que cambia entre entornos: la app la lee y cae de vuelta
a la ruta local del checkpoint cuando no está definida, así que exactamente el
mismo archivo corre en un laptop y en el Space.

El modelo desplegado es BETO y no mBERT. Los dos son estadísticamente
indistinguibles (ΔF1 = 0,0015, p = 0,75) y, ante un empate, el modelo específico
para español es el mejor default para entrada en español: un vocabulario más
chico y mejor ajustado significa menos sub-tokens por palabra e inferencia más
rápida en CPU.

---

## Estructura del repositorio

```
spanish-ner-benchmark/
├── configs/default.yaml        cada hiperparámetro que afecta una métrica
├── data/raw/                   archivos del corpus (en gitignore, con checksum)
├── src/spanish_ner/
│   ├── data.py                 parser CoNLL, esquema de etiquetas, solapamiento
│   ├── baselines.py            gazetteer y CRF
│   ├── modeling.py             alineación de sub-tokens, inferencia transformer
│   ├── evaluate.py             scoring con seqeval, bootstrap pareado
│   └── utils.py                seeding, config, reportes en JSON
├── scripts/
│   ├── download_data.py        descarga y verifica el corpus
│   ├── train_baselines.py      gazetteer + CRF, evaluación en dev
│   ├── train_transformer.py    fine-tuning de cualquier checkpoint de HF
│   ├── evaluate_test.py        el único script que lee el split de test
│   └── error_analysis.py       figuras y categorización de errores
├── serve/
│   ├── api.py                  servicio FastAPI con sondas y métricas
│   ├── benchmark.py            medición de latencia, throughput y costo
│   └── Dockerfile              multi-stage, non-root, solo CPU
├── tests/                      60 tests
├── reports/                    métricas en JSON, figuras en PNG
└── app/                        demo Gradio para Hugging Face Spaces
```

---

## Limitaciones y próximos pasos

Dichas sin rodeos, porque cada una de ellas es una pregunta que vale la pena que
te hagan.

**Dominio.** Anecdóticamente el modelo transfiere mejor de lo que sugieren los
datos de entrenamiento: etiqueta correctamente `Gabriel Boric` y `Rosanna Costa`
como PER y `Banco Central de Chile` como ORG, nada de lo cual puede aparecer en
un corpus en español del año 2000. Eso es alentador, y **no es evidencia**. Son
cuatro oraciones elegidas a mano, sin anotaciones gold detrás.

El corpus es cable noticioso en español de mayo de 2000, de una sola agencia
(EFE), y fuertemente europeo en vocabulario y topónimos. El desempeño sobre texto
chileno, redes sociales, notas clínicas o documentos legales será
sustancialmente más bajo y **no está medido aquí**. Cualquier número de este
README es una afirmación sobre cables de EFE y nada más.

**MISC es débil.** F1 = 0,671, contra 0,958 en PER. La clase está definida en
negativo en las guías de anotación, así que no tiene una forma superficial
consistente. Mejorarla probablemente necesita una formulación distinta, no más
datos de entrenamiento.

**Una sola seed.** Cada configuración se entrenó una vez con la seed 42. El
bootstrap cuantifica la incertidumbre de la *muestra de test*, no la varianza
entre corridas de entrenamiento. La varianza por seed en fine-tuning de BERT
sobre un corpus de este tamaño suele ser de unas pocas décimas de punto de F1.
Comparable a la brecha BETO/mBERT, lo que refuerza el hallazgo 2 en lugar de
debilitarlo. Entrenar cada modelo con cinco seeds y reportar media ± desviación
estándar es el siguiente paso correcto, y no está hecho aquí.

**Sin búsqueda de hiperparámetros.** La tasa de aprendizaje, el tamaño de batch y
el número de épocas son valores estándar de la literatura de fine-tuning de BERT,
aplicados de forma idéntica a los tres transformers. Justo para comparar, casi
con certeza no óptimo para ninguno.

**Truncamiento.** `max_length=256` sub-tokens. Ninguna oración de entrenamiento
se ve afectada, así que esto no cuesta nada en este corpus, pero la demo etiqueta
en silencio como `O` las palabras que pasan ese límite. Los documentos largos
deberían dividirse en trozos antes de pasarlos.

**La inversión de ranking dev/test merece más trabajo.** El hallazgo 1 es una
sola observación. Determinar si XLM-R es genuinamente peor aquí o simplemente tuvo mala
suerte requiere corridas repetidas.

### Planificado

- Entrenamiento multi-seed con reporte de varianza
- Evaluación sobre español chileno para medir directamente la brecha de dominio
- Una capa de decodificación CRF sobre el transformer, que debería reducir el
  38,8 % de errores de límite
- Calibración de confianza, para que la demo pueda marcar predicciones inciertas

---

## Cita

```bibtex
@inproceedings{tjongkimsang2002conll,
  title     = {Introduction to the CoNLL-2002 Shared Task: Language-Independent
               Named Entity Recognition},
  author    = {Tjong Kim Sang, Erik F.},
  booktitle = {Proceedings of CoNLL-2002},
  year      = {2002}
}

@inproceedings{canete2020spanish,
  title     = {Spanish Pre-Trained BERT Model and Evaluation Data},
  author    = {Ca{\~n}ete, Jos{\'e} and Chaperon, Gabriel and Fuentes, Rodrigo and
               Ho, Jou-Hui and Kang, Hojin and P{\'e}rez, Jorge},
  booktitle = {PML4DC at ICLR 2020},
  year      = {2020}
}
```

## Licencia

**MIT, ver [LICENSE](LICENSE).** El corpus CoNLL-2002 lo distribuye la
Universidad de Amberes bajo sus propios términos.
