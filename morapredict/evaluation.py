"""Evaluación offline: etiquetas posteriores se unen solo después de predecir."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from .data import KEY, TARGET, check


def top_k(frame, scores, cupo):
    check(isinstance(cupo,int) and 0<=cupo<=len(frame), 'Cupo fuera de cohorte')
    ranked=frame[KEY].copy();ranked['score']=np.asarray(scores)
    ranked=ranked.sort_values(['score','cliente_id','credito_id'],ascending=[False,True,True])
    return frame.index.isin(ranked.index[:cupo]).astype(int)


def metrics(y, pred):
    y=np.asarray(y).astype(bool);pred=np.asarray(pred).astype(bool)
    tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
    return {'clientes':len(y),'eventos':int(y.sum()),'alertas':int(pred.sum()),'eventos_detectados':tp,
        'falsos_positivos':fp,'falsos_negativos':fn,'verdaderos_negativos':tn,
        'precision':tp/(tp+fp) if tp+fp else None,'sensibilidad':tp/(tp+fn) if tp+fn else None,
        'tasa_falsos_positivos':fp/(fp+tn) if fp+tn else None}


def evaluate(frame, labels, scores, cupo=425):
    audit=frame.merge(labels,on=KEY,validate='1:1',how='left',suffixes=('','_futuro'))
    check(audit[TARGET].notna().all(), 'Faltan etiquetas en evaluación')
    check(audit[KEY].equals(frame[KEY].reset_index(drop=True)), 'Orden de evaluación alterado')
    records=[];segments=[];budget_rows=[];lead=[];prob=[];predictions={}
    for method,score in scores.items():
        pred=top_k(frame,score,cupo);predictions[method]=pred
        m={'metodo':method,**metrics(audit[TARGET],pred)}
        positives=audit[TARGET].eq(1).to_numpy();selected=pred.astype(bool)&positives
        days_due=(audit.proximo_vencimiento-audit.fecha_corte).dt.days
        days_event=(audit.fecha_alcanza_mora_7dias-audit.fecha_corte).dt.days
        m['anticipacion_mediana_dias_al_vencimiento']=float(days_due[selected].median()) if selected.any() else None
        m['anticipacion_mediana_dias_al_evento']=float(days_event[selected].median()) if selected.any() else None
        # Intervalos descriptivos de bootstrap pareado, sin cambiar entrenamiento ni cupo.
        rng=np.random.default_rng(20261120);boots=[]
        for _ in range(500):
            ix=rng.integers(0,len(audit),len(audit));boots.append(metrics(audit[TARGET].to_numpy()[ix],pred[ix]))
        for metric in ['precision','sensibilidad']:
            vals=[b[metric] for b in boots if b[metric] is not None]
            m[metric+'_ic95_inferior'],m[metric+'_ic95_superior']=np.quantile(vals,[.025,.975]).tolist()
        records.append(m)
        for dimension in ['sector','zona_sintetica']:
            for group,indices in audit.groupby(dimension).groups.items():
                ix=np.asarray(list(indices));segments.append({'metodo':method,'dimension':dimension,'grupo':group,
                    **metrics(audit[TARGET].iloc[ix],pred[ix])})
        for k in sorted({50,100,200,425, min(600,len(audit))}):
            if k<=len(audit):budget_rows.append({'metodo':method,'cupo':k,**metrics(audit[TARGET],top_k(frame,score,k))})
        for idx in np.flatnonzero(selected):
            lead.append({'metodo':method,'cliente_id':audit.iloc[idx].cliente_id,
                'fecha_corte':audit.iloc[idx].fecha_corte,'proximo_vencimiento':audit.iloc[idx].proximo_vencimiento,
                'fecha_evento':audit.iloc[idx].fecha_alcanza_mora_7dias,
                'dias_aviso_al_vencimiento':int(days_due.iloc[idx]),'dias_aviso_al_evento':int(days_event.iloc[idx])})
        if method!='Regla':
            prob.append({'metodo':method,'roc_auc':roc_auc_score(audit[TARGET],score),
                'precision_promedio':average_precision_score(audit[TARGET],score),
                'brier_sin_calibrar':brier_score_loss(audit[TARGET],score)})
    return {'comparacion':pd.DataFrame(records),'segmentos':pd.DataFrame(segments),
            'cupos':pd.DataFrame(budget_rows),'anticipacion':pd.DataFrame(lead),
            'probabilidad':pd.DataFrame(prob),'predicciones':predictions}
