# alert-triage-ml

Agrupa logs parecidos com K-Means e sinaliza alertas possivelmente ruidosos com classificação supervisionada.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: holdout de logs agrupado por template — silhouette cosine **0,7701** (175 logs); holdout da classificação de ruído — acurácia **0,8533**, F1 **0,8667**, ROC AUC **0,9319** (300 alertas). O rótulo “ruidoso” é derivado de regras sintéticas, então as métricas não medem incidentes reais.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
