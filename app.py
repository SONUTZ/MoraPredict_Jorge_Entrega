"""Interfaz para procesar cada entrega completa y consultar sus alertas."""
import json
import hashlib
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from morapredict.workflow import (FILES, read_folder, read_uploads, fingerprint,
                                 completed_run, process_delivery, read_run)

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'models'
RUNS=ROOT/'runs'
st.set_page_config(page_title='MoraPredict AI',layout='wide')
st.title('MoraPredict AI')
st.caption('Procesamiento de cartera completa · Alertas preventivas con evidencia')
st.info('La base inicial es sintética. El score requiere revisión humana y no ejecuta decisiones crediticias.')


@st.cache_resource
def load_result(path,signature):
    # Recursos de solo lectura; la ficha se copia antes de modificar su cupo.
    return read_run(path)


original=ROOT.parents[1]/'Data_Deterioro_Pago_Anticipado_5000_Ampliada'
default_folder=original if original.is_dir() else ROOT/'data'
st.subheader('1. Entrega de datos')
method=st.radio('Origen de la entrega',['Carpeta local','Cargar CSV o ZIP'],horizontal=True)
source=None;files=None;input_error=None
if method=='Carpeta local':
    source=st.text_input('Carpeta con los archivos de la entrega',value=str(default_folder))
    try:files=read_folder(source)
    except (OSError,ValueError) as exc:input_error=str(exc)
else:
    uploaded=st.file_uploader('Cargue los cuatro CSV o un ZIP que los contenga',
                             type=['csv','zip'],accept_multiple_files=True)
    if uploaded:
        try:files=read_uploads([(f.name,f.getvalue()) for f in uploaded])
        except (OSError,ValueError) as exc:input_error=str(exc)
        source=', '.join(f.name for f in uploaded)
st.caption('Archivos necesarios: '+', '.join(FILES)+'. No se necesitan resultados futuros ni archivos de partición.')
process_clicked=st.button('Procesar entrega completa',type='primary',disabled=files is None)
if input_error:st.error(input_error)
if files is None:
    st.info('Seleccione una carpeta válida o cargue los archivos para iniciar el proceso.')
    st.stop()
batch_hash=fingerprint(files)
model_hash=hashlib.sha256((MODELS/'xgboost.json').read_bytes()).hexdigest()
current=st.session_state.get('current_delivery')
if current and (current['fingerprint']!=batch_hash or current.get('model_hash')!=model_hash):
    st.session_state.pop('current_delivery',None)
    current=None
if process_clicked:
    message=st.empty()
    try:
        with st.spinner('Validando datos, calculando señales y generando evidencia para toda la cartera...'):
            result_path=process_delivery(files,MODELS,RUNS,source=source,progress=message.write)
        st.session_state['current_delivery']={'fingerprint':batch_hash,'model_hash':model_hash,'path':str(result_path)}
        st.session_state.pop('review_budget',None)
        current=st.session_state['current_delivery']
        st.success('Entrega completa procesada. Los resultados y los archivos fuente quedaron guardados juntos.')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        st.session_state.pop('current_delivery',None)
        st.error('La entrega no se publicó: '+str(exc))
        st.stop()
if current is None:
    existing=completed_run(files,MODELS,RUNS)
    if existing:
        current={'fingerprint':batch_hash,'model_hash':model_hash,'path':str(existing)}
        st.session_state['current_delivery']=current
    else:
        st.warning('Esta entrega es nueva o cambió. Pulse Procesar entrega completa. No se muestran alertas antiguas.')
        st.stop()
result_path=Path(current['path'])
signature=(result_path/'manifest.json').stat().st_mtime_ns
alerts,analytic,series,payments,manifest=load_result(str(result_path),signature)
total=len(alerts)
st.caption(f"Entrega: {manifest['run_id']} · Corte: {manifest['fecha_corte']} · Origen: {source}")
tab1,tab2=st.tabs(['Cartera y evidencia','Calidad de la entrega'])
with tab1:
    st.subheader(f"2. Cartera completa: {total:,} clientes".replace(',','.'))
    cupo=st.slider('Cupo de revisión sobre la cartera completa',0,total,
                   min(manifest['cupo_solicitado'],total),1,key='review_budget')
    st.caption('Todos los clientes reciben puntaje. El cupo solo determina cuántos se seleccionan para revisión.')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Clientes procesados',f'{total:,}'.replace(',','.'))
    c2.metric('Seleccionados para revisión',str(cupo))
    c3.metric('Observaciones semanales',f'{len(series):,}'.replace(',','.'))
    c4.metric('Cuotas históricas',f'{len(payments):,}'.replace(',','.'))
    ranking=pd.DataFrame([{k:r[k] for k in ['cliente_id','prioridad','score','numero_senales_regla']} for r in alerts])
    ranking=ranking.merge(analytic[['cliente_id','negocio_sintetico','sector','zona_sintetica']],on='cliente_id',validate='1:1')
    ranking['seleccionado']=ranking.prioridad.le(cupo)
    only_selected=st.checkbox('Mostrar solo clientes dentro del cupo',value=False)
    visible=ranking[ranking.seleccionado] if only_selected else ranking
    st.caption(f'Se muestran {len(visible):,} de {total:,} clientes.'.replace(',','.'))
    st.dataframe(visible[['prioridad','cliente_id','negocio_sintetico','sector','zona_sintetica','score','numero_senales_regla','seleccionado']],
                 hide_index=True,width='stretch')
    st.download_button('Descargar ranking de la entrega',ranking.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f"ranking_{manifest['fecha_corte']}_{total}_clientes.csv",mime='text/csv')
    st.subheader('3. Ficha del cliente')
    selected=st.selectbox('Cliente para revisar',[r['cliente_id'] for r in alerts])
    alert=next(r for r in alerts if r['cliente_id']==selected).copy()
    alert.update(cupo=cupo,seleccionado=alert['prioridad']<=cupo,
                 estado_revision='Pendiente' if alert['prioridad']<=cupo else 'Sin asignar')
    row=analytic[analytic.cliente_id.eq(selected)].iloc[0]
    col1,col2,col3,col4=st.columns(4)
    col1.metric('Score estimado',f"{alert['score']:.3f}")
    col2.metric('Prioridad en cartera',f"{alert['prioridad']} / {total:,}".replace(',','.'))
    col3.metric('Estado al corte','Al día')
    col4.metric('Dentro del cupo','Sí' if alert['seleccionado'] else 'No')
    st.write(f"**{row.negocio_sintetico} · {row.sector} · {row.zona_sintetica} · corte {alert['fecha_corte']}**")
    st.write(alert['resumen'])
    if alert['calidad']['variables_faltantes']:
        st.warning('Variables faltantes: '+', '.join(alert['calidad']['variables_faltantes']))
    s=series[series.cliente_id.eq(selected)].sort_values('fin_semana')
    p=payments[payments.cliente_id.eq(selected)].sort_values('fecha_vencimiento').copy()
    p['dias_relativos_al_vencimiento']=(p.fecha_pago-p.fecha_vencimiento).dt.days
    a,b=st.columns(2)
    a.plotly_chart(px.line(s,x='fin_semana',y=['depositos_7d_soles','ventas_estimadas_7d_soles'],markers=True,
                           title='Depósitos y ventas estimadas (PEN)'),width='stretch')
    b.plotly_chart(px.line(s,x='fin_semana',y=['saldo_promedio_7d_soles','saldo_final_soles'],markers=True,
                           title='Saldos (PEN)'),width='stretch')
    a,b=st.columns(2)
    fig=px.line(s,x='fin_semana',y='uso_linea_credito',markers=True,title='Uso de línea de capital de trabajo')
    fig.update_yaxes(tickformat='.0%');a.plotly_chart(fig,width='stretch')
    b.plotly_chart(px.bar(p,x='fecha_vencimiento',y='dias_relativos_al_vencimiento',
                         title='Cuotas: días relativos al vencimiento'),width='stretch')
    st.caption('Días negativos = pagó antes; cero = al vencimiento; positivos = después.')
    st.subheader('Factores del modelo y evidencia')
    evidence=pd.DataFrame(alert['evidencia'])
    evidence['desde']=evidence.periodo.apply(lambda x:x['inicio'])
    evidence['hasta']=evidence.periodo.apply(lambda x:x['fin'])
    st.dataframe(evidence[['descripcion','valor','unidad','desde','hasta','shap_log_odds','efecto_modelo','dato_faltante']],
                 hide_index=True,width='stretch')
    factors=pd.DataFrame(alert['motivos']).sort_values('shap_log_odds')
    st.plotly_chart(px.bar(factors,x='shap_log_odds',y='descripcion',orientation='h',
                          title='Mayores contribuciones SHAP (log-odds)'),width='stretch')
    st.caption('SHAP explica el modelo, no causas. El score no tiene calibración externa.')
    with st.expander('Ver semanas y cuotas originales'):
        st.dataframe(s,hide_index=True);st.dataframe(p,hide_index=True)
    st.download_button('Descargar ficha JSON',json.dumps(alert,ensure_ascii=False,indent=2),
                       file_name=f'{selected}.json',mime='application/json')
with tab2:
    st.subheader('Control de la entrega procesada')
    st.json(manifest['calidad'])
    st.write('Cada entrega se valida y puntúa completa, sin separar clientes de entrenamiento y prueba.')
    st.write('Se conserva una copia de los cuatro archivos, la base analítica, las alertas y el ranking de esta ejecución.')
    st.download_button('Descargar control de calidad',json.dumps(manifest,ensure_ascii=False,indent=2),
                       file_name='control_entrega.json',mime='application/json')
