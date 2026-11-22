"""Procesa una entrega completa desde carpeta; sin etiquetas ni particiones."""
import argparse
from pathlib import Path
from morapredict.workflow import read_folder,process_delivery


def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description='MoraPredict: procesar toda una nueva entrega')
    p.add_argument('--data',default=str(root/'data'))
    p.add_argument('--models',default=str(root/'models'))
    p.add_argument('--out',default=str(root/'runs'))
    p.add_argument('--cupo',type=int,default=425)
    args=p.parse_args()
    result=process_delivery(read_folder(args.data),args.models,args.out,args.cupo,
                            source=str(Path(args.data).resolve()),progress=lambda text:print(text,flush=True))
    print('Resultados: '+str(result.resolve()),flush=True)


if __name__=='__main__':main()
