"""Gerard: ejecutar desde el proyecto. No carga resultados futuros."""
from pathlib import Path
from morapredict.workflow import read_folder,process_delivery,read_run

ROOT=Path(__file__).resolve().parent
carpeta=process_delivery(read_folder(ROOT/'data'),ROOT/'models',ROOT/'runs',cupo=425)
cohorte,base,semanas,pagos,control=read_run(carpeta)
seleccionadas=[a for a in cohorte if a['seleccionado']]
print(f'{len(cohorte)} clientes puntuados; {len(seleccionadas)} seleccionados')
print(seleccionadas[0]['resumen'])
# En FastAPI, devolver seleccionadas o recuperar ficha por alerta_id.
# El backend de Gerard resuelve autenticación, permisos y persistencia de gestiones.
