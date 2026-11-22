"""Preparación desde fuentes al corte. Los resultados futuros se leen por separado."""
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['cliente_id', 'credito_id', 'fecha_corte']
PAIR = KEY[:2]
TARGET = 'alcanzo_mora_7dias_en_30d'
BASE_FEATURES = [
    'saldo_promedio_7d_soles', 'variacion_depositos_7d', 'variacion_depositos_4s',
    'variacion_saldo_4s', 'pendiente_saldo_4s', 'uso_linea_credito',
    'variacion_uso_linea_4s', 'cobertura_proxima_cuota', 'flujo_neto_7d_soles',
    'cuotas_atrasadas_historicas', 'max_dias_atraso_historico', 'dias_hasta_vencimiento',
]
PERSIST_FEATURES = [f for tag in ['depositos', 'ventas', 'transacciones'] for f in
                    (f'variacion_{tag}_4s_vs_mediana_8s', f'semanas_{tag}_bajo_80pct')]
PAY_FEATURES = ['proporcion_cuotas_puntuales', 'dias_relativos_pago_primeras3',
                'dias_relativos_pago_ultimas3', 'cambio_dias_relativos_pago',
                'dias_desde_ultimo_pago', 'saldo_promedio_28d_soles']
REFERENCE_FEATURES = BASE_FEATURES + PERSIST_FEATURES
FEATURES = REFERENCE_FEATURES + PAY_FEATURES


class DataValidationError(ValueError):
    """No se permite calcular alertas sobre una base inconsistente."""


def check(condition, message):
    if not bool(condition):
        raise DataValidationError(message)


def read_csv(path):
    return pd.read_csv(path, encoding='utf-8-sig', dtype={
        'cliente_id': str, 'credito_id': str, 'cuota_id': str})


def load_inputs(root):
    """No abre etiquetas, predicciones previas ni archivos SOLO_EVALUACION."""
    return {name: read_csv(Path(root) / f'{name}.csv') for name in
            ['clientes', 'creditos', 'series_semanales', 'historial_pagos']}


def dates(frame, columns):
    for c in columns:
        frame[c] = pd.to_datetime(frame[c], errors='raise').dt.normalize()
        check(frame[c].notna().all(), f'Fecha obligatoria vacía: {c}')


def validate_inputs(inputs):
    c, cr, s, p = (inputs[k].copy() for k in
                   ['clientes', 'creditos', 'series_semanales', 'historial_pagos'])
    dates(cr, ['fecha_corte', 'proximo_vencimiento'])
    dates(s, ['inicio_semana', 'fin_semana'])
    dates(p, ['fecha_vencimiento', 'fecha_pago'])
    for frame, cols, name in [(c, ['cliente_id'], 'clientes'), (cr, KEY, 'creditos'),
                               (s, PAIR + ['fin_semana'], 'series'),
                               (p, ['cuota_id'], 'pagos')]:
        check(frame[cols].notna().all().all(), f'Clave vacía en {name}')
        check(not frame.duplicated(cols).any(), f'Clave duplicada en {name}')
    check(cr.cliente_id.is_unique and cr.credito_id.is_unique,
          'MVP requiere un crédito y un corte por cliente; no agregar créditos sin definir consolidación')
    check(set(c.cliente_id) == set(cr.cliente_id), 'Clientes y créditos no concuerdan')
    check(cr.fecha_corte.nunique() == 1, 'MVP requiere un solo corte por ejecución')
    check(cr.dias_mora_al_corte.eq(0).all(), 'Hay clientes que ya están en mora al corte')
    check((cr.proximo_vencimiento > cr.fecha_corte).all(), 'Vencimiento debe ser posterior al corte')
    check((cr.cuota_mensual_soles > 0).all(), 'Cuota debe ser positiva')
    check((cr.linea_capital_trabajo_soles > 0).all(), 'Límite de línea debe ser positivo')
    for frame, name in [(s, 'series'), (p, 'pagos')]:
        merged = frame.merge(cr[KEY], on=PAIR, how='left', validate='m:1', indicator=True)
        check(merged['_merge'].eq('both').all(), f'Referencia desconocida en {name}')
        cols = ['fin_semana'] if name == 'series' else ['fecha_pago', 'fecha_vencimiento']
        for col in cols:
            check((merged[col] <= merged.fecha_corte).all(), f'Información posterior al corte: {name}.{col}')
        counts = frame.groupby(PAIR).size()
        expected = 12 if name == 'series' else 6
        check(len(counts) == len(cr) and counts.eq(expected).all(),
              f'{name}: se requieren {expected} observaciones por crédito')
    s = s.sort_values(PAIR + ['fin_semana']).reset_index(drop=True)
    p = p.sort_values(PAIR + ['fecha_vencimiento']).reset_index(drop=True)
    check((s.fin_semana - s.inicio_semana).dt.days.eq(6).all(), 'Semana no tiene siete días')
    gaps = s.groupby(PAIR).fin_semana.diff().dropna().dt.days
    check(gaps.eq(7).all(), 'Semanas no consecutivas')
    last = s.groupby(PAIR).fin_semana.max().reset_index().merge(cr[KEY], on=PAIR)
    check(last.fin_semana.eq(last.fecha_corte).all(), 'Última semana no termina en el corte')
    numeric = ['saldo_inicial_soles','depositos_7d_soles','egresos_7d_soles',
               'saldo_final_soles','saldo_promedio_7d_soles','uso_linea_credito',
               'ventas_estimadas_7d_soles','numero_transacciones','proporcion_ventas_digitales']
    check(not np.isinf(s[numeric].to_numpy(dtype=float)).any(), 'Valores infinitos en series')
    check(s[numeric].dropna().ge(0).all().all(), 'Importes o actividad negativos')
    for col in ['uso_linea_credito','proporcion_ventas_digitales']:
        check(s[col].dropna().between(0, 1).all(), f'{col} fuera de [0,1]')
    balance_cols = ['saldo_inicial_soles','depositos_7d_soles','egresos_7d_soles','saldo_final_soles']
    complete = s[balance_cols].notna().all(axis=1)
    residual = s.saldo_inicial_soles + s.depositos_7d_soles - s.egresos_7d_soles - s.saldo_final_soles
    check(residual[complete].abs().le(.011).all(), 'Saldos semanales no concilian')
    prev = s.groupby(PAIR).saldo_final_soles.shift()
    available = prev.notna() & s.saldo_inicial_soles.notna()
    check((s.loc[available,'saldo_inicial_soles']-prev[available]).abs().le(.011).all(),
          'Continuidad de saldos entre semanas inválida')
    late = (p.fecha_pago - p.fecha_vencimiento).dt.days.clip(lower=0)
    check(late.eq(p.dias_atraso_observado).all(), 'Atrasos históricos inconsistentes')
    check((p.monto_exigible_soles > 0).all() and
          (p.monto_pagado_soles-p.monto_exigible_soles).abs().le(.011).all(),
          'MVP requiere cuotas históricas completamente pagadas')
    check(not p.duplicated(PAIR+['fecha_vencimiento']).any(), 'Vencimientos históricos duplicados')
    return c, cr, s, p, {
        'clientes': len(c), 'creditos': len(cr), 'series': len(s), 'pagos': len(p),
        'fecha_corte': cr.fecha_corte.iloc[0].date().isoformat(),
        'celdas_faltantes_series': int(s[numeric].isna().sum().sum()),
        'conciliaciones_omitidas_por_faltantes': int((~complete).sum()),
        'checks': ['claves y referencias','mora cero','fechas al corte','12 semanas consecutivas',
                   '6 cuotas pagadas','rangos','conciliación y continuidad de saldos']}


def ratio(a, b):
    return float(a / b) if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def strict_mean(a):
    return float(a.mean()) if a.notna().all() else np.nan


def build_features(inputs):
    """Una fila por cliente/crédito/corte. Recalcula; no confía en variables prehechas."""
    c, cr, s, p, audit = validate_inputs(inputs)
    series = dict(tuple(s.groupby(PAIR)))
    payments = dict(tuple(p.groupby(PAIR)))
    rows = []
    for credit in cr.to_dict('records'):
        pair = (credit['cliente_id'], credit['credito_id'])
        w = series[pair].reset_index(drop=True)
        h = payments[pair].reset_index(drop=True)
        last, old = w.iloc[-1], w.iloc[-5]
        signed = (h.fecha_pago - h.fecha_vencimiento).dt.days
        record = {k: credit[k] for k in KEY}
        record.update({
            'saldo_promedio_7d_soles': last.saldo_promedio_7d_soles,
            'variacion_depositos_7d': ratio(last.depositos_7d_soles, w.iloc[-2].depositos_7d_soles)-1,
            'variacion_depositos_4s': ratio(strict_mean(w.depositos_7d_soles.iloc[-4:]),
                                            strict_mean(w.depositos_7d_soles.iloc[-8:-4]))-1,
            'variacion_saldo_4s': ratio(last.saldo_promedio_7d_soles,
                                       max(1., old.saldo_promedio_7d_soles) if pd.notna(old.saldo_promedio_7d_soles) else np.nan)-1,
            'pendiente_saldo_4s': float(np.polyfit(np.arange(4), w.saldo_promedio_7d_soles.iloc[-4:], 1)[0])
                if w.saldo_promedio_7d_soles.iloc[-4:].notna().all() else np.nan,
            'uso_linea_credito': last.uso_linea_credito,
            'variacion_uso_linea_4s': last.uso_linea_credito - old.uso_linea_credito,
            'cobertura_proxima_cuota': ratio(last.saldo_final_soles, credit['cuota_mensual_soles']),
            'flujo_neto_7d_soles': last.depositos_7d_soles - last.egresos_7d_soles,
            'cuotas_atrasadas_historicas': int(signed.gt(0).sum()),
            'max_dias_atraso_historico': int(signed.clip(lower=0).max()),
            'dias_hasta_vencimiento': (credit['proximo_vencimiento']-credit['fecha_corte']).days,
            'dias_mora_actual': credit['dias_mora_al_corte'],
            'proporcion_cuotas_puntuales': signed.le(0).mean(),
            'dias_relativos_pago_primeras3': signed.iloc[:3].mean(),
            'dias_relativos_pago_ultimas3': signed.iloc[-3:].mean(),
            'cambio_dias_relativos_pago': signed.iloc[-3:].mean()-signed.iloc[:3].mean(),
            'dias_desde_ultimo_pago': (credit['fecha_corte']-h.fecha_pago.max()).days,
            'saldo_promedio_28d_soles': strict_mean(w.saldo_promedio_7d_soles.iloc[-4:]),
            'depositos_promedio_reciente_soles': strict_mean(w.depositos_7d_soles.iloc[-4:]),
            'depositos_promedio_anterior_soles': strict_mean(w.depositos_7d_soles.iloc[-8:-4]),
            'uso_linea_4semanas_antes': old.uso_linea_credito,
            'saldo_final_reciente_soles': last.saldo_final_soles,
            'cuota_mensual_soles': credit['cuota_mensual_soles'],
            'proximo_vencimiento': credit['proximo_vencimiento'],
        })
        for column, tag in [('depositos_7d_soles','depositos'),
                            ('ventas_estimadas_7d_soles','ventas'),('numero_transacciones','transacciones')]:
            history, recent = w[column].iloc[:8], w[column].iloc[-4:]
            baseline = history.median() if history.notna().all() else np.nan
            record[f'variacion_{tag}_4s_vs_mediana_8s'] = ratio(strict_mean(recent), baseline)-1
            record[f'semanas_{tag}_bajo_80pct'] = int(recent.lt(.8*baseline).sum()) \
                if pd.notna(baseline) and baseline > 0 and recent.notna().all() else np.nan
        signals = [record['variacion_depositos_4s'],record['pendiente_saldo_4s'],
                   record['variacion_uso_linea_4s'],record['cobertura_proxima_cuota']]
        record['numero_senales'] = sum([signals[0] <= -.2,signals[1] < 0,signals[2] >= .1,signals[3] < 1])
        record['alerta_preventiva_regla'] = int(record['numero_senales'] >= 3) if all(pd.notna(signals)) else np.nan
        record['senales_regla_incompletas'] = not all(pd.notna(signals))
        record['variables_faltantes'] = ','.join(f for f in FEATURES if pd.isna(record[f]))
        rows.append(record)
    analytic = pd.DataFrame(rows).merge(c, on='cliente_id', validate='1:1')
    analytic[FEATURES] = analytic[FEATURES].round(6)
    audit['filas_analiticas'] = len(analytic)
    audit['filas_con_variables_faltantes'] = int(analytic[FEATURES].isna().any(axis=1).sum())
    return analytic.sort_values(KEY).reset_index(drop=True), audit


def load_labels(root, analytic):
    """Reconstruye el evento futuro; jamás devuelve predictores futuros."""
    y = read_csv(Path(root) / 'SOLO_EVALUACION_resultados_30d.csv')
    dates(y, ['fecha_corte','fin_horizonte','fecha_vencimiento'])
    for col in ['fecha_pago_observada_30d','fecha_alcanza_mora_7dias']:
        y[col] = pd.to_datetime(y[col], errors='raise')
    check(not y.duplicated(KEY).any(), 'Etiquetas duplicadas')
    expected_keys = analytic[KEY+['proximo_vencimiento']].merge(y, on=KEY, how='outer',
                                                              validate='1:1', indicator=True)
    check(expected_keys['_merge'].eq('both').all(), 'Etiquetas no cubren exactamente la base')
    check(expected_keys.proximo_vencimiento.eq(expected_keys.fecha_vencimiento).all(), 'Vencimiento futuro inconsistente')
    check((y.fin_horizonte-y.fecha_corte).dt.days.eq(30).all(), 'Horizonte distinto de 30 días')
    paid = y.fecha_pago_observada_30d
    check((paid.isna() | ((paid > y.fecha_corte) & (paid <= y.fin_horizonte))).all(), 'Pago futuro fuera de horizonte')
    event_date = y.fecha_vencimiento + pd.Timedelta(days=7)
    event = (event_date > y.fecha_corte) & (event_date <= y.fin_horizonte) & (paid.isna() | (paid >= event_date))
    check(y[TARGET].isin([0,1]).all() and event.astype(int).eq(y[TARGET]).all(), 'Etiqueta no coincide con evento de mora 7/30')
    check(y.loc[event,'fecha_alcanza_mora_7dias'].eq(event_date[event]).all() and
          y.loc[~event,'fecha_alcanza_mora_7dias'].isna().all(), 'Fecha del evento inconsistente')
    y['evento_evaluable_en_horizonte'] = (event_date <= y.fin_horizonte).astype(int)
    return y
