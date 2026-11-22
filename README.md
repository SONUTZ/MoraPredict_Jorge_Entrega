# MoraPredict AI — Procesar cada nueva entrega de datos

La aplicación valida los archivos, calcula las señales y genera puntajes y evidencia para **todos los clientes de la entrega**. La base principal contiene 5.000 clientes, 60.000 observaciones semanales y 30.000 cuotas históricas. No se divide la cartera durante este proceso.

## Usar la interfaz

1. Abra la aplicación y elija **Carpeta local** o **Cargar CSV o ZIP**.
2. Indique la carpeta de la nueva entrega, cargue los cuatro CSV o un ZIP que los contenga.
3. Pulse **Procesar entrega completa**.
4. Consulte la cartera completa, ajuste el cupo de revisión, abra una ficha o descargue el ranking.

Si los archivos cambian, la aplicación deja de mostrar los resultados anteriores hasta procesar la nueva entrega. Una entrega idéntica puede reutilizar una ejecución anterior únicamente después de verificar sus archivos y su modelo. Los originales no se modifican. Cada ejecución conserva su propia copia de datos y resultados.

## Archivos de entrada

- `clientes.csv`: cliente, nombre del negocio, sector, zona y antigüedad.
- `creditos.csv`: cliente, crédito, fecha de corte, próximo vencimiento, cuota, límite de línea y mora al corte.
- `series_semanales.csv`: saldos, depósitos, egresos, ventas estimadas, transacciones y uso de línea.
- `historial_pagos.csv`: cuotas, vencimientos, fechas e importes pagados.

Use los nombres y columnas de los CSV de ejemplo en `data/`. Formato UTF-8 con BOM opcional, separador coma, punto decimal y fechas ISO `yyyy-mm-dd`. Los identificadores se conservan como texto. Los demás archivos del ZIP original no son necesarios ni se usan para producir las alertas.

El motor actual requiere un crédito por cliente, un corte por entrega, 12 semanas consecutivas que terminen en el corte y seis cuotas históricas pagadas. Atiende clientes que están al día al corte. Valida estas condiciones y no publica entregas inconsistentes. El total de clientes se toma de los archivos: la entrega principal se procesa con sus 5.000 clientes completos y una nueva entrega puede tener otro tamaño sin modificar el código.

## Ejecutar

Python 3.12 recomendado. Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python score.py --data data --out runs
python -m streamlit run app.py --global.developmentMode false
```

Para procesar una carpeta directamente sin interfaz:

```powershell
python score.py --data "C:\ruta\nueva_entrega" --out runs --cupo 425
```

No necesita ejecutar entrenamiento ni entregar resultados futuros para este proceso.

## Archivos de salida

Cada entrega completada se guarda en una carpeta propia dentro de `runs/`:

- `inputs/`: copia de las cuatro tablas utilizadas.
- `base_analitica.csv`: señales recalculadas para todos los clientes.
- `ranking.csv`: score, prioridad y selección de revisión para toda la cartera.
- `alertas.json`: motivos, valores, fechas y contribuciones SHAP por cliente.
- `calidad.json`: resultado de las validaciones.
- `manifest.json`: huella de datos, versión del modelo, fecha de corte, conteos y hashes de integridad.

`runs/latest.json` apunta a la última ejecución completada. Una entrega inválida no reemplaza ese resultado. Las carpetas `.procesando-*` incompletas no se muestran como resultados válidos.

El cupo de revisión inicial es 425 y se puede ajustar. Todos los clientes se puntúan; el cupo decide cuántos se revisan primero. Al cambiar el cupo en la interfaz, el CSV descargado y la ficha reflejan esa selección. Los archivos persistidos de una ejecución conservan el cupo con el que se procesó; la línea de comandos permite guardar una ejecución con otro cupo.

## Integración en Python

```python
from morapredict.workflow import read_folder, process_delivery, read_run

archivos = read_folder('nueva_entrega')
carpeta_resultado = process_delivery(archivos, 'models', 'runs', cupo=425)
alertas, base, semanas, pagos, control = read_run(carpeta_resultado)
pendientes = [a for a in alertas if a['seleccionado']]
```

`morapredict/data.py` recalcula las variables desde las fuentes. `morapredict/engine.py` carga el modelo y calcula el ranking y SHAP. `morapredict/workflow.py` coordina la recepción, validación y publicación de cada entrega. No requiere API, proveedor de IA externo ni credenciales.

## Modelo y límites

La operación usa el modelo ya entrenado. **No se vuelve a entrenar con cada entrega** ni se supone que conoce sus resultados futuros. Reentrenar requiere eventos posteriores observados y una validación técnica independiente. Los archivos del desarrollo inicial que se conserven son históricos y no participan en la interfaz ni en `score.py`.

Las contribuciones SHAP están en log-odds y describen la salida del modelo; no son causas ni puntos porcentuales de probabilidad. El score no está calibrado externamente. La base inicial y el entrenamiento son sintéticos: aplicar el prototipo a una cartera real requiere validar su adecuación. Las alertas requieren revisión humana y no ejecutan decisiones crediticias.

## Pruebas

Primero procese los datos sintéticos de ejemplo. Las ejecuciones se generan localmente y no se versionan en Git.

```powershell
python score.py --data data --out runs
python -m pytest tests -q
```

Las pruebas verifican entrega sin etiquetas, recalculado ante cambios, conservación de ejecuciones anteriores, rechazo de datos inválidos, CSV/ZIP equivalentes, integridad de resultados, cartera principal completa, SHAP y controles de la interfaz.
