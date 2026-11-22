"""Inferencia reutilizable. No requiere etiquetas ni acceso a resultados futuros."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from .data import KEY, FEATURES, check


def native(value):
    if isinstance(value, dict):
        return {str(k):native(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(native(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def dictionary():
    """Fórmulas, ventanas y unidades. Las fechas concretas se agregan por cliente."""
    d = {
        'saldo_promedio_7d_soles': ('Saldo promedio última semana','PEN','última semana','Media de 7 saldos diarios (fuente semanal)'),
        'variacion_depositos_7d': ('Variación depósitos semanal','ratio','últimas 2 semanas','depósitos última semana / anterior - 1; vacío si denominador <= 0'),
        'variacion_depositos_4s': ('Variación depósitos 4 semanas','ratio','últimas 8 semanas','media últimas 4 / media 4 anteriores - 1; vacío si base <= 0'),
        'variacion_saldo_4s': ('Variación saldo en 4 semanas','ratio','semana actual y 4 semanas antes','saldo promedio actual / max(1 PEN, saldo promedio semana -4) - 1'),
        'pendiente_saldo_4s': ('Tendencia saldo','PEN/semana','últimas 4 semanas','Pendiente OLS de 4 saldos promedio contra índices [0,1,2,3]'),
        'uso_linea_credito': ('Uso actual de línea','ratio','última semana','Fracción utilizada de la línea de capital de trabajo'),
        'variacion_uso_linea_4s': ('Cambio uso de línea','ratio; ×100 = puntos porcentuales','semana actual y 4 semanas antes','uso actual - uso semana -4'),
        'cobertura_proxima_cuota': ('Cobertura próxima cuota','veces','cierre del corte','saldo final última semana / cuota mensual positiva'),
        'flujo_neto_7d_soles': ('Flujo neto semanal','PEN','última semana','depósitos - egresos'),
        'cuotas_atrasadas_historicas': ('Cuotas históricas atrasadas','cuotas','6 cuotas históricas','Conteo de fecha_pago > fecha_vencimiento'),
        'max_dias_atraso_historico': ('Máximo atraso histórico','días','6 cuotas históricas','max(0, fecha_pago - vencimiento)'),
        'dias_hasta_vencimiento': ('Días hasta próxima cuota','días','corte a próximo vencimiento','próximo vencimiento - corte'),
        'proporcion_cuotas_puntuales': ('Proporción de cuotas puntuales','ratio','6 cuotas históricas','media(fecha_pago <= vencimiento)'),
        'dias_relativos_pago_primeras3': ('Días relativos pago primeras 3','días','primeras 3 cuotas','media(fecha_pago - vencimiento); negativos = anticipación'),
        'dias_relativos_pago_ultimas3': ('Días relativos pago últimas 3','días','últimas 3 cuotas','media(fecha_pago - vencimiento); negativos = anticipación'),
        'cambio_dias_relativos_pago': ('Cambio de puntualidad','días','6 cuotas históricas','media días relativos últimas 3 - media primeras 3; positivo = pago más tardío'),
        'dias_desde_ultimo_pago': ('Días desde último pago','días','último pago a corte','corte - max(fecha_pago conocida)'),
        'saldo_promedio_28d_soles': ('Saldo promedio 28 días','PEN','últimas 4 semanas','media de los 4 saldos promedio semanales; semanas de igual duración'),
    }
    for tag in ['depositos','ventas','transacciones']:
        d[f'variacion_{tag}_4s_vs_mediana_8s'] = (f'Variación persistente {tag}','ratio','12 semanas',
            'media últimas 4 / mediana primeras 8 - 1; vacío si mediana <= 0 o hay faltantes')
        d[f'semanas_{tag}_bajo_80pct'] = (f'Semanas bajas {tag}','semanas','12 semanas',
            'Conteo últimas 4 < 0.8 × mediana primeras 8; vacío si mediana <= 0 o hay faltantes')
    return d


def feature_period(row, feature, history=None):
    end = pd.Timestamp(row['fecha_corte'])
    window = dictionary()[feature][2]
    days = {'última semana':6,'últimas 2 semanas':13,'últimas 8 semanas':55,
            'últimas 4 semanas':27,'12 semanas':83,'semana actual y 4 semanas antes':34,
            'cierre del corte':0}
    if window in days:
        return {'inicio':(end-pd.Timedelta(days=days[window])).date().isoformat(), 'fin':end.date().isoformat()}
    if feature == 'dias_hasta_vencimiento':
        return {'inicio':end.date().isoformat(),'fin':pd.Timestamp(row['proximo_vencimiento']).date().isoformat()}
    if feature == 'dias_desde_ultimo_pago':
        return {'inicio':(end-pd.Timedelta(days=int(row[feature]))).date().isoformat(),'fin':end.date().isoformat()}
    if history is not None and not history.empty:
        h=history
        if feature.endswith('primeras3'): h=h.iloc[:3]
        if feature.endswith('ultimas3'): h=h.iloc[-3:]
        return {'inicio':pd.to_datetime(h.fecha_vencimiento).min().date().isoformat(),
                'fin':max(pd.to_datetime(h.fecha_vencimiento).max(),pd.to_datetime(h.fecha_pago).max()).date().isoformat()}
    return {'inicio':None,'fin':end.date().isoformat()}


class MotorAlertas:
    """Carga un Booster JSON y genera un ranking con SHAP exacto en log-odds."""
    def __init__(self, model, metadata):
        self.model, self.metadata = model, metadata
        check(metadata['features'] == FEATURES, 'Esquema de variables incompatible')

    @classmethod
    def load(cls, directory):
        directory=Path(directory)
        metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        check(hashlib.sha256((directory/'xgboost.json').read_bytes()).hexdigest()==metadata['sha256_modelo'],
              'Modelo alterado o versión incorrecta')
        booster=xgb.Booster()
        booster.load_model(directory/'xgboost.json')
        return cls(booster, metadata)

    def generate(self, analytic, cupo=425, payments=None):
        """Orden estable: score descendente, luego cliente_id/credito_id. No usa etiquetas.

        cupo se aplica a una cohorte de un único corte. Para una ficha aislada,
        recuperar el resultado de la cohorte; no interpretar ranking=1 como prioridad global.
        """
        check(isinstance(cupo, int) and cupo >= 0, 'Cupo debe ser entero no negativo')
        required=KEY+FEATURES+['dias_mora_actual','proximo_vencimiento','numero_senales','variables_faltantes']
        check(set(required).issubset(analytic.columns), 'Faltan columnas de la base analítica')
        a=analytic.copy()
        check(not a.empty and not a.duplicated(KEY).any(), 'Base vacía o claves duplicadas')
        check(a.fecha_corte.nunique()==1, 'No mezclar fechas de corte en el ranking')
        check(a.dias_mora_actual.eq(0).all(), 'Motor preventivo solo admite clientes al día')
        X=a[self.metadata['features']].astype(float)
        check(not np.isinf(X.to_numpy()).any(), 'Predictores infinitos')
        check(not X.isna().all(axis=1).any(), 'Cliente sin ninguna señal: no se permite puntuar')
        dm=xgb.DMatrix(X, feature_names=self.metadata['features'], missing=np.nan)
        score=self.model.predict(dm)
        shap=self.model.predict(dm,pred_contribs=True,approx_contribs=False)
        margin=self.model.predict(dm,output_margin=True)
        check(np.allclose(shap.sum(axis=1),margin,atol=2e-5), 'SHAP no reconcilia con el margen')
        check(np.allclose(1/(1+np.exp(-shap.sum(axis=1))),score,atol=2e-6), 'SHAP no reconcilia con score')
        a['_row']=np.arange(len(a));a['score']=score
        a=a.sort_values(['score','cliente_id','credito_id'],ascending=[False,True,True]).reset_index(drop=True)
        records=[];defs=dictionary()
        histories={}
        if payments is not None:
            p=payments.copy()
            for date_col in ['fecha_vencimiento','fecha_pago']:
                p[date_col]=pd.to_datetime(p[date_col],errors='raise')
                check(p[date_col].notna().all() and (p[date_col]<=pd.Timestamp(a.fecha_corte.iloc[0])).all(),
                      'Historial explicativo contiene fechas posteriores al corte o faltantes')
            histories=dict(tuple(p.sort_values('fecha_vencimiento').groupby(['cliente_id','credito_id'])))
        for rank, row in enumerate(a.to_dict('records'),1):
            i=row['_row'];all_factors=[]
            h=histories.get((row['cliente_id'],row['credito_id']))
            for j,f in enumerate(self.metadata['features']):
                label,unit,window,formula=defs[f]
                all_factors.append({'variable':f,'descripcion':label,'valor':native(row[f]),'unidad':unit,
                    'ventana':window,'periodo':feature_period(row,f,h),'formula':formula,
                    'fuente':'historial_pagos.csv' if f in ['cuotas_atrasadas_historicas','max_dias_atraso_historico',
                        'proporcion_cuotas_puntuales','dias_relativos_pago_primeras3','dias_relativos_pago_ultimas3',
                        'cambio_dias_relativos_pago','dias_desde_ultimo_pago'] else
                        'creditos.csv' if f=='dias_hasta_vencimiento' else
                        'series_semanales.csv + creditos.csv' if f=='cobertura_proxima_cuota' else 'series_semanales.csv',
                    'shap_log_odds':float(shap[i,j]),'efecto_modelo':'aumenta' if shap[i,j]>0 else 'reduce' if shap[i,j]<0 else 'neutro',
                    'dato_faltante':bool(pd.isna(row[f]))})
            factors=sorted(all_factors,key=lambda v:abs(v['shap_log_odds']),reverse=True)
            cutoff=pd.Timestamp(row['fecha_corte']).date().isoformat()
            ident=hashlib.sha256(f"{row['cliente_id']}|{row['credito_id']}|{cutoff}|{self.metadata['version']}".encode()).hexdigest()[:20]
            selected=rank<=min(cupo,len(a))
            records.append(native({'alerta_id':ident,'cliente_id':row['cliente_id'],'credito_id':row['credito_id'],
                'fecha_corte':cutoff,'version_modelo':self.metadata['version'],'version_datos':self.metadata['version_datos'],
                'prioridad':rank,'seleccionado':selected,'cupo':cupo,'tamano_cohorte':len(a),'score':float(row['score']),
                'tipo_score':'estimación sintética sin calibración externa','estado_revision':'Pendiente' if selected else 'Sin asignar',
                'dias_mora_actual':0,'numero_senales_regla':row['numero_senales'],
                'calidad':{'variables_faltantes':[f for f in FEATURES if pd.isna(row[f])],
                           'tratamiento':'XGBoost maneja ausencias; la evidencia conserva nulos'},
                'motivos':factors[:5],'evidencia':all_factors,'shap_base_log_odds':float(shap[i,-1]),
                'shap_margen_total':float(margin[i]),'resumen':describe(row),
                'accion_sugerida':'Revisar evidencia y considerar contacto preventivo; requiere criterio humano',
                'decision_crediticia_automatica':False}))
        return records


def describe(row):
    pieces=['Cliente al día al corte.']
    d=row['variacion_depositos_4s']
    if pd.notna(d):
        pieces.append(f"Los depósitos promedio de las últimas 4 semanas {'bajaron' if d<0 else 'subieron'} {abs(d)*100:.1f}% frente a las 4 anteriores.")
    if pd.notna(row.get('uso_linea_4semanas_antes')) and pd.notna(row['uso_linea_credito']):
        pieces.append(f"Uso de línea: {row['uso_linea_4semanas_antes']*100:.1f}% a {row['uso_linea_credito']*100:.1f}% en 4 semanas.")
    if pd.notna(row['cobertura_proxima_cuota']):
        pieces.append(f"El saldo final cubre {row['cobertura_proxima_cuota']*100:.1f}% de la próxima cuota.")
    if pd.notna(row.get('cambio_dias_relativos_pago')):
        pieces.append(f"Cambio de días relativos al vencimiento, últimas 3 cuotas frente a primeras 3: {row['cambio_dias_relativos_pago']:+.1f} días.")
    return ' '.join(pieces)
