# Kompilierbare LaTeX-Fassung

Startdatei: `main.tex`. XeLaTeX oder Tectonic verwenden; pdfLaTeX ist wegen fontspec ungeeignet. Erforderliche Systemschriften: Times New Roman, Arial und Consolas. Die geprüfte Windows-Umgebung enthält diese Schriften. Alternativ in einer passenden TeX-Umgebung bereitstellen; Schriftwechsel verändern das Layout.

```powershell
tectonic -X compile main.tex --keep-logs
```

Alternativ XeLaTeX mindestens zweimal ausführen, bis Verzeichnisse und Verweise stabil sind. Es wird keine BibTeX-Datei benötigt: Das mitgelieferte Original nutzte ein manuelles Literaturverzeichnis. Die geprüfte Fassung führt dieses in `bibliography.tex` weiter. Die Beispiel-BibTeX-Datei einer Formatvorlage war keine Literaturdatenbank dieser Arbeit.

KI-Erklärung und ehrenwörtliche Erklärung müssen vor Einreichung vom Verfasser geprüft, datiert und unterschrieben werden. Die Unterschriften wurden nicht erzeugt. Maßgeblich ist die tatsächlich unterzeichnete Einreichungsfassung.
