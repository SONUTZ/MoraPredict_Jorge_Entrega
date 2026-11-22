import io
import json
import zipfile
from pathlib import Path
import pandas as pd
import pytest
from morapredict.data import DataValidationError,load_inputs
from morapredict.workflow import (FILES, fingerprint, read_folder, read_uploads,
                                 process_delivery,completed_run,read_run)

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def files():
    raw=load_inputs(ROOT/'data')
    ids=['CLI-00001','CLI-00002','CLI-00003']
    return {name:raw[Path(name).stem].loc[lambda d:d.cliente_id.isin(ids)].to_csv(index=False).encode('utf-8-sig')
            for name in FILES}


def test_only_four_inputs_no_future_labels_required(files,tmp_path):
    data=tmp_path/'data';data.mkdir()
    for name,content in files.items():(data/name).write_bytes(content)
    path=process_delivery(read_folder(data),ROOT/'models',tmp_path/'runs',2)
    alerts,base,series,payments,manifest=read_run(path)
    assert len(alerts)==len(base)==3 and len(series)==36 and len(payments)==18
    assert sum(a['seleccionado'] for a in alerts)==2
    assert manifest['clientes']==3
    assert all(a['version_datos']==fingerprint(files) for a in alerts)
    assert all('alcanzo_mora_7dias_en_30d' not in a for a in alerts)
    assert completed_run(files,ROOT/'models',tmp_path/'runs',2)==path
    assert process_delivery(files,ROOT/'models',tmp_path/'runs',2)==path


def test_new_delivery_recomputes_features_and_preserves_previous(files,tmp_path):
    old=process_delivery(files,ROOT/'models',tmp_path,1)
    before=read_run(old)
    series=pd.read_csv(io.BytesIO(files['series_semanales.csv']))
    ix=series[series.cliente_id.eq('CLI-00001')].sort_values('fin_semana').tail(4).index
    series.loc[ix,'depositos_7d_soles']+=1000
    series.loc[ix,'egresos_7d_soles']+=1000
    changed={**files,'series_semanales.csv':series.to_csv(index=False).encode('utf-8-sig')}
    assert completed_run(changed,ROOT/'models',tmp_path,1) is None
    new=process_delivery(changed,ROOT/'models',tmp_path,1)
    after=read_run(new)
    assert new!=old and old.is_dir()
    old_value=before[1].set_index('cliente_id').loc['CLI-00001','depositos_promedio_reciente_soles']
    new_value=after[1].set_index('cliente_id').loc['CLI-00001','depositos_promedio_reciente_soles']
    assert new_value-old_value==pytest.approx(1000)
    old_evidence=next(a for a in before[0] if a['cliente_id']=='CLI-00001')['evidencia']
    new_evidence=next(a for a in after[0] if a['cliente_id']=='CLI-00001')['evidencia']
    assert next(f['valor'] for f in old_evidence if f['variable']=='variacion_depositos_4s')!=next(
        f['valor'] for f in new_evidence if f['variable']=='variacion_depositos_4s')


def test_invalid_delivery_does_not_replace_latest(files,tmp_path):
    good=process_delivery(files,ROOT/'models',tmp_path,1)
    previous=(tmp_path/'latest.json').read_bytes()
    s=pd.read_csv(io.BytesIO(files['series_semanales.csv']))
    s=pd.concat([s,s.iloc[:1]])
    bad={**files,'series_semanales.csv':s.to_csv(index=False).encode()}
    with pytest.raises(DataValidationError,match='duplicada'):
        process_delivery(bad,ROOT/'models',tmp_path,1)
    assert (tmp_path/'latest.json').read_bytes()==previous
    assert read_run(good)[4]['clientes']==3


def test_zip_and_csv_upload_equivalent(files):
    content=io.BytesIO()
    with zipfile.ZipFile(content,'w') as z:
        for name,data in files.items():z.writestr('entrega/'+name,data)
        z.writestr('SOLO_EVALUACION_resultados_30d.csv','no se necesita')
    from_zip=read_uploads([('entrega.zip',content.getvalue())])
    assert fingerprint(from_zip)==fingerprint(files)
    assert read_uploads(list(files.items()))==files
    with pytest.raises(DataValidationError,match='repetido'):
        read_uploads([('entrega.zip',content.getvalue()),('clientes.csv',files['clientes.csv'])])
    with pytest.raises(DataValidationError,match='Faltan archivos'):
        read_uploads([('clientes.csv',files['clientes.csv'])])


def test_modified_output_is_not_silently_reused(files,tmp_path):
    old=process_delivery(files,ROOT/'models',tmp_path,1)
    (old/'ranking.csv').write_text('alterado',encoding='utf-8')
    assert completed_run(files,ROOT/'models',tmp_path,1) is None
    new=process_delivery(files,ROOT/'models',tmp_path,1)
    assert new!=old
    assert completed_run(files,ROOT/'models',tmp_path,1)==new


def test_published_main_delivery_contains_all_clients():
    files=read_folder(ROOT/'data')
    path=completed_run(files,ROOT/'models',ROOT/'runs')
    assert path is not None
    alerts,base,series,payments,manifest=read_run(path)
    assert len(alerts)==len(base)==manifest['clientes']==5000
    assert len(series)==60000 and len(payments)==30000
    assert {a['cliente_id'] for a in alerts}==set(pd.read_csv(ROOT/'data/clientes.csv').cliente_id)
    assert sum(a['seleccionado'] for a in alerts)==425
