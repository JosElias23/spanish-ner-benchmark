# Reconocimiento de entidades nombradas en español: de tablas de búsqueda a transformers

[![CI](https://github.com/JosElias23/spanish-ner-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/JosElias23/spanish-ner-benchmark/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-73%20passing-brightgreen)](https://github.com/JosElias23/spanish-ner-benchmark/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[English](README.md) · **Español**

NER de punta a punta para español sobre el benchmark CoNLL-2002. Cinco modelos,
un solo protocolo de evaluación, nada seleccionado sobre el conjunto de test
reservado, e intervalos de confianza en cada comparación — con corrección de Holm
sobre las diez.

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
que hace la mayoría de los tutoriales, habría dejado en producción el modelo que
terminó último en test. Esto es sobreajuste en la selección de modelo sobre un
conjunto de desarrollo de 1.915 oraciones, observado directamente en vez de
descrito en abstracto.

**Lo que el ranking no significa.** Una versión anterior de esta sección llamaba
a XLM-R «el peor de los tres transformers» y decía que más grande y más
multilingüe fue «estrictamente peor en todos los ejes que importan». El ranking
es real; las brechas que lo producen no se distinguen del ruido. Corregidas las
diez comparaciones por multiplicidad (hallazgo 2), ningún par de transformers se
separa en test: XLM-R contra BETO da p = 0,30 y XLM-R contra mBERT, p = 0,13. Lo
honesto es decir que XLM-R lideró dev y terminó último en test por márgenes que
este conjunto de prueba no puede resolver.

Lo que sí queda sin matices es el costo. XLM-R son 277 M de parámetros y 452 s de
entrenamiento frente a los 109 M y 105 s de BETO: 4,3× más caro para un puntaje
que, en el mejor de los casos, es el mismo.

### 2. Un encoder específico para español no da ninguna ventaja medible aquí

Esta es la hipótesis para la que se construyó el proyecto, así que está nombrada
en el código como la comparación confirmatoria y se reporta aparte, sin ajustar:

| Comparación | ΔF1 | IC 95 % | p | Significativo |
|---|---:|:--|---:|:--:|
| BETO vs mBERT | −0,0015 | [−0,0113, +0,0086] | 0,75 | **no** |

**BETO y mBERT son estadísticamente indistinguibles, y en este benchmark la
hipótesis del modelo específico para español no se sostiene.** Reportar
`mBERT 0,8720 > BETO 0,8705` como resultado habría sido una afirmación sobre
ruido de muestreo.

Todo lo demás es exploratorio: los diez pares, en orden alfabético, con
corrección de Holm sobre la familia.

| Comparación | ΔF1 | IC 95 % | p | p (Holm) | Significativo |
|---|---:|:--|---:|---:|:--:|
| BETO vs gazetteer | +0,5110 | [+0,4934, +0,5290] | <0,001 | <0,001 | sí |
| gazetteer vs mBERT | −0,5125 | [−0,5316, −0,4931] | <0,001 | <0,001 | sí |
| gazetteer vs XLM-R | −0,5027 | [−0,5231, −0,4817] | <0,001 | <0,001 | sí |
| CRF vs gazetteer | +0,4329 | [+0,4150, +0,4508] | <0,001 | <0,001 | sí |
| BETO vs CRF | +0,0782 | [+0,0614, +0,0953] | <0,001 | <0,001 | sí |
| CRF vs mBERT | −0,0797 | [−0,0978, −0,0624] | <0,001 | <0,001 | sí |
| CRF vs XLM-R | −0,0698 | [−0,0887, −0,0520] | <0,001 | <0,001 | sí |
| mBERT vs XLM-R | +0,0098 | [+0,0003, +0,0195] | 0,042 | **0,126** | **no** |
| BETO vs XLM-R | +0,0083 | [−0,0028, +0,0207] | 0,151 | 0,303 | no |
| BETO vs mBERT | −0,0015 | [−0,0113, +0,0086] | 0,752 | 0,752 | no |

> **Corrección.** Esta tabla tenía cuatro filas, todas contra mBERT, y mBERT fue
> elegido porque tenía el F1 más alto **en test**:
>
> ```python
> ranked = sorted(results, key=lambda n: results[n]["full"]["overall"]["f1"],
>                 reverse=True)
> for challenger in ranked[1:]:
>     comparisons[f"{ranked[0]}_vs_{challenger}"] = paired_bootstrap(...)
> ```
>
> El brazo de referencia se elegía por su puntaje sobre los mismos datos de los
> que salen los p-valores, lo que sesga cada comparación a su favor, y las
> cuatro pruebas se imprimían como si cada una fuera única y pre-registrada. La
> fila que importaba era **mBERT vs XLM-R con p = 0,042, marcada «sí»**: le gana
> a 0,05 por 0,008 y su intervalo cruza el cero por 0,0003 de F1. Con Holm queda
> en **0,126**, y no es un hallazgo.
>
> La comparación que el protocolo por ranking nunca podía producir es BETO vs
> XLM-R, porque ninguno de los dos era el ganador. Da p = 0,151, y por eso el
> hallazgo 1 ya no llama a XLM-R el peor de nada.
>
> Algo que vale la pena notar en la dirección contraria: el sesgo empujaba *en
> contra* de la conclusión del propio proyecto. mBERT era el brazo que la
> selección favorecía, y aun así BETO no pudo separarse de él. El hallazgo 2 era
> conservador, no favorecido.

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
- **Deduplicado**: las 1.166 oraciones *distintas* ausentes del entrenamiento,
  una estimación más estricta de la generalización a texto genuinamente nuevo.

  Ese número decía 1.208 acá, que es el conteo después de quitar las oraciones
  de test que aparecen en entrenamiento pero antes de quitar las 42 que se
  repiten dentro del propio test. `scripts/evaluate_test.py` siempre descartó
  las dos cosas —una oración de relleno que aparece cuatro veces no debería
  pesar cuatro veces en una medición libre de memorización— pero solo el primer
  paso estaba descrito, así que la prosa nombraba un conjunto 42 oraciones más
  grande que el que efectivamente se puntuaba. `reports/metrics_test.json` ahora
  registra los tres conteos.

El ranking no cambia, pero el *costo* de la deduplicación no es uniforme, y el
patrón dice algo:

| Modelo | Completo | Deduplicado | Caída |
|---|---:|---:|---:|
| Gazetteer | 0,3595 | 0,3364 | **−2,31 pts** |
| CRF | 0,7924 | 0,7769 | −1,55 pts |
| XLM-R | 0,8622 | 0,8530 | −0,92 pts |
| BETO | 0,8705 | 0,8642 | −0,63 pts |
| mBERT | 0,8720 | 0,8659 | −0,61 pts |

La caída está ordenada exactamente por el F1 general, y ahí está el problema con
la lectura que le daba una versión anterior de esta sección.

> **Corrección.** Acá decía: «mientras más se apoya un modelo en la
> memorización, más pierde cuando se quitan las oraciones memorizadas — el
> gazetteer, que no es más que memorización, cede casi cuatro veces más que los
> transformers», y lo llamaba una medición directa de la generalización.
>
> Dos verificaciones lo rompen, y las dos usan solo las cifras que ya están en
> esta página.
>
> **El orden de las caídas es el orden del F1, invertido, exactamente.** La
> correlación de Spearman entre el F1 general en test y la caída, sobre los
> cinco modelos, es **−1,00**. Una correlación de rango perfecta sobre cinco
> modelos es la firma de una restricción aritmética, no de una diferencia de
> comportamiento.
>
> **La restricción es el margen disponible.** Quitar las oraciones duplicadas
> quita el 9,4 % de las entidades de test. Tratando el micro-F1 como
> aproximadamente un promedio ponderado sobre las dos porciones, un modelo que
> puntuara *perfecto* en ese 9,4 % podría caer a lo sumo
> `0,094 × (1 − F1) / 0,906` cuando se lo quitan:
>
> | Modelo | F1 test | Caída máxima que permite la aritmética | Observada |
> |---|---:|---:|---:|
> | Gazetteer | 0,3595 | 6,64 pts | 2,31 |
> | CRF | 0,7924 | 2,15 pts | 1,55 |
> | XLM-R | 0,8622 | 1,43 pts | 0,92 |
> | BETO | 0,8705 | 1,34 pts | 0,63 |
> | mBERT | 0,8720 | 1,33 pts | 0,61 |
>
> La razón entre los techos del gazetteer y de mBERT es **5,0×** antes de que
> ningún modelo se comporte de ninguna manera. La razón observada es **3,8×**,
> por debajo del límite mecánico. Un modelo con puntaje bajo tiene más espacio
> para caer, y estas caídas son compatibles con eso y con nada más.
>
> Lo que la tabla sí muestra es más chico y vale la pena conservarlo: todos los
> modelos pierden algo, así que todos estaban recibiendo crédito por oraciones
> ya vistas en entrenamiento, y la columna deduplicada es la estimación más
> honesta. Lo que no muestra es cuál modelo se apoyó más en la memorización.
> Separar eso exigiría comparar el puntaje de cada modelo *sobre las oraciones
> duplicadas* contra oraciones equivalentes ausentes del entrenamiento, y ese
> experimento no está corrido acá.

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
- **Nada se selecciona sobre el split de test.** La selección de baselines, los
  hiperparámetros y el early stopping corren todos sobre dev, y leer `esp.testb`
  exige pasar `include_test=True`, así que no puede ocurrir por accidente. Dos
  scripts lo pasan: `scripts/evaluate_test.py`, que produce los puntajes
  reportados, y `scripts/error_analysis.py`, que describe los errores del modelo
  ya elegido y no cambió ninguno. Una versión anterior de esta viñeta decía que
  `evaluate_test.py` era el único archivo que leía el split de test, y no era
  cierto.
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

73 tests: integridad del corpus, encoding, solapamiento de splits, decodificación
de entidades, alineación de etiquetas a sub-tokens, el protocolo de
significancia y el contrato del servicio.
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

Medido en una sola RTX 5060 Ti. **Cada fila procesa los mismos 512 documentos**
del conjunto de desarrollo (24,2 tokens en promedio), en trozos del tamaño de
batch indicado, tres pasadas cada una después de cinco calentamientos
descartados. p50/p95/p99 son latencias por batch sobre todos los trozos de todas
las pasadas; el throughput es la carga completa de 512 documentos dividida por la
mediana del tiempo de pasada. Producido por `serve/benchmark.py` y guardado en
[`reports/metrics_serving.json`](reports/metrics_serving.json).

| Batch | p50 (ms) | p95 (ms) | p99 (ms) | Docs/s | USD por 1M de docs |
|---:|---:|---:|---:|---:|---:|
| 1 | 6,4 | 7,1 | 7,6 | 160 | 0,92 |
| 4 | 8,9 | 12,1 | 12,5 | 431 | 0,34 |
| 8 | 13,7 | 20,5 | 22,2 | 551 | 0,27 |
| 16 | 24,5 | 36,7 | 37,2 | 638 | 0,23 |
| **32** | 52,8 | 64,3 | 64,7 | **638** | **0,23** |
| 64 | 112,7 | 127,7 | 128,4 | 605 | 0,24 |

**Hacer batching de 32 documentos entrega 4,0× el throughput de procesarlos de a
uno y recorta el costo por millón de documentos en un 75 %.**

> **Corrección.** Esta tabla decía 6,7× y ponía el óptimo en batch 64, porque
> cada fila se cronometraba sobre `pool[:batch_size]`: batch 1 sobre una oración
> en particular y batch 64 sobre otras 64 distintas. La comparación entre filas
> mezclaba entonces el tamaño de batch con qué documentos se midieron, y la fila
> de batch 1 era la latencia de una sola oración desafortunada en vez de una
> mediana. Con la carga fija, el speed-up es 4,0× y **el throughput hace pico en
> 32 y baja en 64** — así que el consejo anterior de poner un pipeline masivo en
> el extremo derecho de la tabla apuntaba pasado el óptimo.

La forma es el punto. Pasar de batch 1 a batch 4 cuesta 2,6 ms de latencia y casi
triplica el throughput, porque un batch de cuatro apenas llena la GPU. De 16 a 32
el throughput no se mueve nada mientras la latencia se duplica, y en 64 va para
atrás. El dispositivo se satura cerca de 16, y de ahí en adelante el batching
solo compra demora de cola. Un endpoint interactivo va a la izquierda de esta
tabla, un pipeline masivo al medio, y ninguno de los dos números por sí solo
describe el sistema.

**Sobre la columna de costo.** La tarifa es de USD 0,53/hora para una NVIDIA T4 y
el throughput es el de la RTX 5060 Ti de esta máquina, que es bastante más
rápida. Así que la columna es una tarifa multiplicada por un throughput medido,
no una cotización: una T4 real cobraría más por millón de documentos. El desajuste
queda registrado en `reports/metrics_serving.json` como `priced_hardware` y
`measured_hardware`, en vez de dejarlo a que el lector lo infiera. Una versión
anterior de esta sección decía solo «una GPU de inferencia de gama de entrada»,
lo que no daba manera de notarlo.

### Cómo correrlo

```bash
pip install -e ".[serve]"
python serve/benchmark.py                    # reproduce la tabla de GPU de arriba
python serve/benchmark.py --device cpu       # la tabla de CPU, para el contenedor
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

El contenedor es multi-etapa y corre como usuario sin privilegios sobre torch
solo-CPU, lo que mantiene la imagen chica para un free tier.

> **Corrección.** Esta frase terminaba «llegando igual a 100 documentos por
> segundo con batch 64», y **en este repositorio no existía ninguna medición en
> CPU**: `reports/metrics_serving.json` era una única corrida en CUDA.
> `serve/benchmark.py` siempre tuvo una opción `--device cpu` que nunca se había
> usado para producir un reporte comiteado. Ahora sí, en
> [`reports/metrics_serving_cpu.json`](reports/metrics_serving_cpu.json):
>
> | Batch | p50 (ms) | Docs/s | USD por 1M de docs |
> |---:|---:|---:|---:|
> | 1 | 28,7 | 35,0 | 0,79 |
> | **4** | 82,0 | **48,6** | **0,57** |
> | 8 | 162,9 | 47,8 | 0,58 |
> | 16 | 368,7 | 43,2 | 0,64 |
> | 32 | 888,0 | 38,9 | 0,71 |
> | 64 | 2005,8 | 34,5 | 0,80 |
>
> **48,6 documentos por segundo con batch 4, no 100 con batch 64** — y batch 64
> es el peor punto de la tabla, ni mejor que procesar de a uno. El batching vale
> 1,4× en esta CPU contra 4,0× en la GPU, y ese es el hallazgo más útil: la CPU
> ya está limitada por cómputo con batch 1, así que agrupar trabajo compra
> latencia y casi nada de throughput. El despliegue en free tier es viable a unas
> pocas decenas de documentos por segundo, y el número anterior era una
> estimación sin medir que resultó ser tres veces optimista.

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
├── tests/                      73 tests
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

**Truncamiento.** `max_length=256` sub-tokens, y sí muerde. Una versión anterior
de esta viñeta decía «ninguna oración de entrenamiento se ve afectada, así que
esto no cuesta nada en este corpus» — contradicho por los propios reportes
comiteados en este repositorio, que registran **6 oraciones de entrenamiento
truncadas y 1.531 palabras perdidas** para BETO (6 / 1.537 para mBERT, 5 / 1.449
para XLM-R). Seis oraciones de 8.323 suena despreciable hasta que uno nota que
son las largas: 1.531 de 264.715 palabras de entrenamiento, **0,58 %**. Poco,
pero no la nada que se afirmaba. El reporte lo escribe
`scripts/train_transformer.py` y cubre solo el split de entrenamiento — dev y
test no se revisan, así que no se sabe nada del truncamiento ahí. La demo además
etiqueta en silencio como `O` las palabras que pasan el límite; los documentos
largos deberían dividirse en trozos antes de pasarlos.

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
