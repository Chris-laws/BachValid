# Reproduktion und veröffentlichter Umfang

Dieses öffentliche Repository enthält die historischen Ergebnisartefakte, Prüfskripte und 13 gekennzeichneten Notebookkopien. Die vollständigen lokalen Abgabearchive `Reproduktion.zip` und `Quellenarchiv.zip` werden nicht veröffentlicht. Rohdaten, wiedergewonnene Rollen-Metadaten/Embeddings und Literaturvolltexte bleiben lokal. Die bereits berechneten Rohscore-/Neufit-Prüfergebnisse sind unter `review/` enthalten.

Vom Repository-Stamm aus ist die Aggregat-/Statistikprüfung direkt ausführbar:

```powershell
python -m pip install -r Reproduktion/requirements.txt
python scripts/empirical_audit.py
```

Erwartet werden 364 bestandene Prüfungen. `raw_score_audit.py` und `closure_audit.py` benötigen zusätzlich `artifacts/raw_recovery/` aus dem vollständigen lokalen Reproduktionspaket. Ohne diese Inputs können ihre historischen Resultate eingesehen, aber nicht neu berechnet werden. `core_report.py` erzeugt das Kernprüfprotokoll aus den gespeicherten Prüf-CSVs.

Die Originalnotebooks dokumentieren die historische Kaggle-/GPU-Umgebung und können zusätzliche ursprüngliche Checkpoints benötigen. Die 13 Notebookkopien und ihre Rollen sind im [Notebookregister](notebooks/README.md) beschrieben. Unbekannte frühere KI-Modellversionen wurden nicht ergänzt.
