"""Proceso repetible de entregas completas, sin particiones ni etiquetas futuras."""
import hashlib
import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import pandas as pd
from .data import DataValidationError, build_features, check
from .engine import MotorAlertas, write_json

FILES=('clientes.csv','creditos.csv','series_semanales.csv','historial_pagos.csv')
MAX_BYTES=120*1024*1024
REQUIRED={
    'clientes.csv':['cliente_id','negocio_sintetico','sector','zona_sintetica','antiguedad_negocio_meses'],
    'creditos.csv':['cliente_id','credito_id','fecha_corte','proximo_vencimiento','dias_mora_al_corte',
                    'cuota_mensual_soles','linea_capital_trabajo_soles'],
    'series_semanales.csv':['cliente_id','credito_id','inicio_semana','fin_semana','saldo_inicial_soles',
        'depositos_7d_soles','egresos_7d_soles','saldo_final_soles','saldo_promedio_7d_soles',
        'uso_linea_credito','ventas_estimadas_7d_soles','numero_transacciones','proporcion_ventas_digitales'],
    'historial_pagos.csv':['cliente_id','credito_id','cuota_id','fecha_vencimiento','fecha_pago',
                         'monto_exigible_soles','monto_pagado_soles','dias_atraso_observado'],
}


def read_folder(folder):
    """Lee solo las cuatro fuentes; ignora resultados futuros o archivos auxiliares."""
    folder=Path(str(folder).strip().strip('"')).expanduser()
    check(folder.is_dir(),f'No se encontró la carpeta: {folder}')
    missing=[name for name in FILES if not (folder/name).is_file()]
    check(not missing,'Faltan archivos: '+', '.join(missing))
    check(sum((folder/n).stat().st_size for n in FILES)<=MAX_BYTES,'La entrega supera 120 MB')
    return {name:(folder/name).read_bytes() for name in FILES}


def read_uploads(uploaded):
    """Lee CSV o ZIP en memoria, sin extraer rutas al disco."""
    files={}
    for filename,content in uploaded:
        check(len(content)<=MAX_BYTES,'Archivo demasiado grande')
        if str(filename).lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    relevant=[i for i in z.infolist() if not i.is_dir() and PurePosixPath(i.filename).name in FILES]
                    check(sum(i.file_size for i in relevant)<=MAX_BYTES,'Contenido del ZIP supera 120 MB')
                    for info in relevant:
                        name=PurePosixPath(info.filename).name
                        check(name not in files,'Archivo repetido en la entrega: '+name)
                        files[name]=z.read(info)
            except (zipfile.BadZipFile,RuntimeError) as exc:
                raise DataValidationError('ZIP inválido o protegido; cargue un ZIP sin contraseña') from exc
        else:
            name=Path(filename).name
            if name in FILES:
                check(name not in files,'Archivo repetido en la entrega: '+name)
                files[name]=content
    validate_files(files)
    return files


def validate_files(files):
    check(set(FILES).issubset(files),'Faltan archivos: '+', '.join(n for n in FILES if n not in files))
    check(sum(len(files[n]) for n in FILES)<=MAX_BYTES,'La entrega supera 120 MB')


def fingerprint(files):
    validate_files(files)
    h=hashlib.sha256()
    for name in FILES:
        h.update(name.encode());h.update(hashlib.sha256(files[name]).digest())
    return h.hexdigest()


def parse_inputs(files):
    validate_files(files);inputs={}
    for name in FILES:
        try:
            df=pd.read_csv(io.BytesIO(files[name]),encoding='utf-8-sig',
                           dtype={'cliente_id':str,'credito_id':str,'cuota_id':str})
        except (ValueError,UnicodeError,pd.errors.ParserError) as exc:
            raise DataValidationError(f'{name}: CSV inválido; use UTF-8, separador coma y punto decimal') from exc
        check(not df.empty,f'{name} está vacío')
        missing=sorted(set(REQUIRED[name])-set(df.columns))
        check(not missing,f'{name}: faltan columnas '+', '.join(missing))
        inputs[Path(name).stem]=df
    return inputs


def delivery_id(files,models,cupo):
    check(isinstance(cupo,int) and cupo>=0,'Cupo debe ser un entero no negativo')
    return f"{fingerprint(files)[:20]}-{hashlib.sha256((Path(models)/'xgboost.json').read_bytes()).hexdigest()[:12]}-c{cupo}"


def completed_run(files,models,out,cupo=425):
    """Busca únicamente una ejecución íntegra de la misma entrega y modelo."""
    run_id=delivery_id(files,models,cupo)
    path=Path(out)/run_id
    latest=Path(out)/'latest.json'
    if latest.is_file():
        try:
            pointer=json.loads(latest.read_text(encoding='utf-8'))
            name=pointer['run_id']
            if Path(name).name==name and name.startswith(run_id+'-reproceso-'):
                path=Path(out)/name
        except (OSError,ValueError,KeyError):pass
    manifest=path/'manifest.json'
    if not manifest.is_file():return None
    try:
        m=json.loads(manifest.read_text(encoding='utf-8'))
        if m['fingerprint_datos']!=fingerprint(files) or m['cupo_solicitado']!=cupo:return None
        if m['modelo_sha256']!=hashlib.sha256((Path(models)/'xgboost.json').read_bytes()).hexdigest():return None
        for name,digest in m['sha256_salidas'].items():
            p=path/name
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:return None
        return path
    except (ValueError,KeyError,OSError):return None


def process_delivery(files,models,out,cupo=425,source='Entrega de datos',progress=None):
    """Recalcula y puntúa TODA la entrega. Solo reutiliza la misma huella verificada.

    Publica una carpeta indivisible y un puntero latest.json únicamente si todo pasó.
    Una entrega inválida no reemplaza la anterior ni obtiene alertas antiguas.
    """
    if progress:progress('Validando archivos de la entrega')
    model=MotorAlertas.load(models)
    existing=completed_run(files,models,out,cupo)
    if existing:return existing
    inputs=parse_inputs(files)
    if progress:progress('Validando clientes, fechas y saldos; recalculando señales')
    analytic,quality=build_features(inputs)
    data_hash=fingerprint(files)
    # La versión de datos pertenece a ESTA entrega, no a la muestra de entrenamiento.
    model=MotorAlertas(model.model,{**model.metadata,'version_datos':data_hash})
    if progress:progress(f'Calculando puntajes y evidencia para {len(analytic):,} clientes')
    alerts=model.generate(analytic,cupo,payments=inputs['historial_pagos'])
    check(len(alerts)==len(analytic) and {a['cliente_id'] for a in alerts}==set(analytic.cliente_id),
          'El resultado no cubre todos los clientes de la entrega')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    run_id=delivery_id(files,models,cupo);final=out/run_id
    # Si una ejecución previa del mismo ID está dañada, conservarla para diagnóstico.
    if final.exists():final=out/(run_id+'-reproceso-'+uuid.uuid4().hex[:8])
    stage=out/('.procesando-'+uuid.uuid4().hex)
    stage.mkdir();(stage/'inputs').mkdir()
    try:
        for name in FILES:(stage/'inputs'/name).write_bytes(files[name])
        analytic.to_csv(stage/'base_analitica.csv',index=False,encoding='utf-8-sig',float_format='%.9f')
        write_json(stage/'alertas.json',alerts)
        cols=['alerta_id','cliente_id','credito_id','fecha_corte','prioridad','seleccionado','score',
              'numero_senales_regla','version_modelo','version_datos','resumen']
        ranking=pd.DataFrame([{k:a[k] for k in cols} for a in alerts]).merge(
            analytic[['cliente_id','negocio_sintetico','sector','zona_sintetica']],on='cliente_id',validate='1:1')
        ranking.to_csv(stage/'ranking.csv',index=False,encoding='utf-8-sig')
        write_json(stage/'calidad.json',quality)
        hashes={p.relative_to(stage).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in stage.rglob('*') if p.is_file()}
        manifest={'run_id':final.name,'fingerprint_datos':data_hash,
            'modelo_sha256':model.metadata['sha256_modelo'],'version_modelo':model.metadata['version'],
            'fecha_corte':quality['fecha_corte'],'procesado_utc':datetime.now(timezone.utc).isoformat(),
            'origen':str(source),'clientes':len(analytic),'cupo_solicitado':cupo,
            'alertas_seleccionadas':sum(a['seleccionado'] for a in alerts),
            'proceso':'Inferencia de toda la entrega; sin partición ni etiquetas futuras',
            'sha256_salidas':hashes,'calidad':quality}
        write_json(stage/'manifest.json',manifest)
        stage.rename(final)
        pointer=out/('.latest-'+uuid.uuid4().hex+'.json')
        write_json(pointer,{'run_id':final.name,'fingerprint_datos':data_hash})
        pointer.replace(out/'latest.json')
    except Exception:
        # Se preservan los archivos parciales para diagnóstico, sin publicarlos como resultado.
        raise
    if progress:progress('Proceso completado: cartera completa y evidencia guardadas')
    return final


def read_run(path):
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    alerts=json.loads((path/'alertas.json').read_text(encoding='utf-8'))
    analytic=pd.read_csv(path/'base_analitica.csv')
    inputs={Path(n).stem:pd.read_csv(path/'inputs'/n) for n in FILES}
    s=inputs['series_semanales'];p=inputs['historial_pagos']
    for f in ['inicio_semana','fin_semana']:s[f]=pd.to_datetime(s[f])
    for f in ['fecha_vencimiento','fecha_pago']:p[f]=pd.to_datetime(p[f])
    check(len(alerts)==manifest['clientes']==len(analytic),'Ejecución incompleta')
    return alerts,analytic,s,p,manifest
