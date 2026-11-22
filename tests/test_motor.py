import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from morapredict import load_inputs, build_features, load_labels, MotorAlertas, DataValidationError
from morapredict.data import FEATURES, KEY, TARGET
from morapredict.evaluation import top_k

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def raw():
    return load_inputs(ROOT/'data')


@pytest.fixture(scope='session')
def base(raw):
    return build_features(raw)[0]


@pytest.fixture(scope='session')
def motor():
    return MotorAlertas.load(ROOT/'models')


def one(raw):
    return {k:v[v.cliente_id.eq('CLI-00001')].copy() for k,v in raw.items()}


def test_population_and_single_grain(base):
    assert len(base)==5000 and not base.duplicated(KEY).any()
    assert base.dias_mora_actual.eq(0).all()
    assert base.fecha_corte.nunique()==1
    assert not base[FEATURES].isna().any().any()


def test_known_first_client(base,raw):
    row=base[base.cliente_id.eq('CLI-00001')].iloc[0]
    assert row.variacion_depositos_4s==pytest.approx(-.291696,abs=1e-6)
    assert row.variacion_uso_linea_4s==pytest.approx(.219159,abs=1e-6)
    assert row.cobertura_proxima_cuota==0
    assert row.numero_senales==4
    p=raw['historial_pagos'].query("cliente_id=='CLI-00001'")
    assert len(p)==6
    assert len(raw['series_semanales'].query("cliente_id=='CLI-00001'"))==12


def test_payment_sign_and_comparison(raw):
    r=one(raw)
    p=r['historial_pagos'].sort_values('fecha_vencimiento').copy()
    due=pd.to_datetime(p.fecha_vencimiento)
    relative=np.array([-5,-4,-3,-1,0,2])
    p['fecha_pago']=(due+pd.to_timedelta(relative,unit='D')).dt.strftime('%Y-%m-%d')
    p['dias_atraso_observado']=np.maximum(relative,0)
    r['historial_pagos']=p
    b,_=build_features(r);row=b.iloc[0]
    assert row.dias_relativos_pago_primeras3==-4
    assert row.dias_relativos_pago_ultimas3==pytest.approx(1/3,abs=1e-6)
    assert row.cambio_dias_relativos_pago==pytest.approx(4+1/3,abs=1e-6)
    assert row.proporcion_cuotas_puntuales==pytest.approx(5/6,abs=1e-6)


def test_future_sources_rejected(raw):
    r=one(raw);idx=r['historial_pagos'].index[-1]
    r['historial_pagos'].loc[idx,'fecha_pago']='2026-11-21'
    with pytest.raises(DataValidationError,match='posterior al corte'):
        build_features(r)


def test_duplicate_and_broken_balance_rejected(raw):
    r=one(raw);r['series_semanales']=pd.concat([r['series_semanales'],r['series_semanales'].iloc[:1]])
    with pytest.raises(DataValidationError,match='duplicada'):
        build_features(r)
    r=one(raw);r['series_semanales'].iloc[0,r['series_semanales'].columns.get_loc('saldo_final_soles')]+=1
    with pytest.raises(DataValidationError,match='concilian'):
        build_features(r)


def test_zero_denominator_is_missing_not_infinite(raw):
    r=one(raw);s=r['series_semanales'].sort_values('fin_semana').copy()
    # Solo se ponen en cero ventas; no altera la conciliación de la cuenta.
    s.loc[s.index[:8],'ventas_estimadas_7d_soles']=0
    r['series_semanales']=s
    b,_=build_features(r)
    assert pd.isna(b.iloc[0].variacion_ventas_4s_vs_mediana_8s)
    assert pd.isna(b.iloc[0].semanas_ventas_bajo_80pct)
    assert 'variacion_ventas_4s_vs_mediana_8s' in b.iloc[0].variables_faltantes
    assert not np.isinf(b[FEATURES].to_numpy()).any()


def test_label_boundaries_and_post_event_payment(tmp_path):
    # Misma fecha de vencimiento; pago día 3 no es evento, día 7 sí, día 9 sí.
    keys=[{'cliente_id':str(i),'credito_id':str(i),'fecha_corte':pd.Timestamp('2026-11-20'),
           'proximo_vencimiento':pd.Timestamp('2026-11-25')} for i in range(4)]
    a=pd.DataFrame(keys);label=[]
    for i,(paid,event) in enumerate([('2026-11-28',0),('2026-12-02',1),('2026-12-04',1),(None,1)]):
        label.append({**{k:keys[i][k] for k in KEY},'fecha_vencimiento':'2026-11-25',
            'fin_horizonte':'2026-12-20','fecha_pago_observada_30d':paid,TARGET:event,
            'fecha_alcanza_mora_7dias':'2026-12-02' if event else None})
    pd.DataFrame(label).to_csv(tmp_path/'SOLO_EVALUACION_resultados_30d.csv',index=False)
    y=load_labels(tmp_path,a)
    assert y[TARGET].tolist()==[0,1,1,1]
    # Un vencimiento demasiado tardío no alcanza 7 días en el horizonte.
    a['proximo_vencimiento']=pd.Timestamp('2026-12-19')
    for l in label:
        l.update(fecha_vencimiento='2026-12-19',fecha_pago_observada_30d=None,
                 fecha_alcanza_mora_7dias=None,alcanzo_mora_7dias_en_30d=0)
    pd.DataFrame(label).to_csv(tmp_path/'SOLO_EVALUACION_resultados_30d.csv',index=False)
    y=load_labels(tmp_path,a)
    assert y[TARGET].eq(0).all() and y.evento_evaluable_en_horizonte.eq(0).all()


def test_predictions_ignore_targets_and_ids_not_features(base,motor,raw):
    sample=base.iloc[:12].copy()
    original=motor.generate(sample,4,payments=raw['historial_pagos'])
    sample[TARGET]=1;sample['dias_mora_al_dia30']=999
    sample['score_previo']=1;sample['sector']='No usar'
    injected=motor.generate(sample,4,payments=raw['historial_pagos'])
    assert [a['score'] for a in original]==[a['score'] for a in injected]
    assert not set(KEY+[TARGET,'numero_senales','alerta_preventiva_regla','sector','zona_sintetica'])&set(FEATURES)


def test_shap_additivity_periods_and_strict_json(base,motor,raw):
    alerts=motor.generate(base.iloc[:8],3,payments=raw['historial_pagos'])
    assert sum(a['seleccionado'] for a in alerts)==3
    for a in alerts:
        total=sum(f['shap_log_odds'] for f in a['evidencia'])+a['shap_base_log_odds']
        assert total==pytest.approx(a['shap_margen_total'],abs=2e-5)
        assert 1/(1+np.exp(-total))==pytest.approx(a['score'],abs=2e-6)
        for f in a['evidencia']:
            assert f['periodo']['inicio'] is not None
            assert f['periodo']['fin']<=a['fecha_corte'] or f['variable']=='dias_hasta_vencimiento'
        json.dumps(a,allow_nan=False)


def test_missing_values_explained_and_no_all_missing_score(base,motor):
    sample=base.iloc[:2].copy();sample.loc[sample.index[0],'variacion_depositos_4s']=np.nan
    result=motor.generate(sample,1)
    a=next(x for x in result if x['cliente_id']==sample.iloc[0].cliente_id)
    f=next(x for x in a['evidencia'] if x['variable']=='variacion_depositos_4s')
    assert f['valor'] is None and f['dato_faltante']
    sample[FEATURES]=np.nan
    with pytest.raises(DataValidationError,match='sin ninguna señal'):
        motor.generate(sample)


def test_stable_order_and_budget(base,motor):
    sample=base.iloc[:10].copy()
    one=motor.generate(sample,0);two=motor.generate(sample.sample(frac=1,random_state=2),999)
    assert [a['cliente_id'] for a in one]==[a['cliente_id'] for a in two]
    assert not any(a['seleccionado'] for a in one)
    assert all(a['seleccionado'] for a in two)
    tied=top_k(sample,np.zeros(len(sample)),2)
    assert tied.tolist()==[1,1]+[0]*8
    with pytest.raises(DataValidationError):motor.generate(sample,-1)


def test_demo_executes_and_client_selection():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
    assert not app.exception
    assert [t.label for t in app.tabs]==['Cartera y evidencia','Calidad de la entrega']
    assert len(app.selectbox[0].options)==5000
    assert len(app.dataframe[0].value)==5000
    assert app.slider[0].max==5000
    app.selectbox[0].select(app.selectbox[0].options[1]).run()
    assert not app.exception
    app.slider[0].set_value(0).run()
    assert not app.exception
    assert len(app.dataframe[0].value)==5000
    app.checkbox[0].check().run()
    assert not app.exception and len(app.dataframe[0].value)==0
    app.slider[0].set_value(5000).run()
    assert not app.exception and len(app.dataframe[0].value)==5000
