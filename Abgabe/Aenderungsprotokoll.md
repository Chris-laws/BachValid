# Kompaktes Änderungsprotokoll

- Vollständiges LaTeX-Projekt aus dem vorhandenen Gesamttext rekonstruiert; Bilder eingebunden, Präambel und Querverweise ergänzt. Das manuelle Literaturverzeichnis bleibt erhalten; eine nicht vorhandene Thesis-BibTeX-Datenbank wurde nicht vorgetäuscht.
- Zentrale Empirie unabhängig neu gerechnet: sämtliche FF1-Differenzen, Interaktionen, N10-Mittel/Intervalle, Sign-Flip, Holm, FF3-Nichtunterlegenheit und Margensensitivität, B95, beide Putra-AP-AULC-Vergleiche, Systemabstände und Routing.
- B95-Aussage korrigiert: frühere relative RUT-Sättigung gilt auf DEV; auf FINAL haben R0/RUT gleiche B95-Budgets je Probe.
- Negatives Ergebnis der zusätzlichen Putra-Zielanpassung ergänzt; positive Quell-RUT-Übertragung und weitere Zielanpassung in Diskussion und Fazit getrennt.
- Labelbudget, Klassenbalance des 200k-Vollbudgets, Datenrollen, Datumsgrenze, fehlende Domainwerte, feste TAPT-Checkpoints und nachträgliche Analyseblöcke präzisiert; 75 % nur auf Downstream-Trainingslabels bezogen.
- P≥90 korrekt als maximale Precision bei Recall mindestens 90 % definiert; FPR@TPR90 als erster entsprechender ROC-Punkt beschrieben. Base-Rate-Formel und vier gerundete Werte korrigiert.
- FF4-Makromittel von Official-Populationsraten getrennt; eingefrorene Kaskade vom N5-Frontendprotokoll unterschieden. Float32-Routinggrenzen, Benchmarkumfang, p95, Streuung und Speicherwerte dokumentiert.
- Alle empirischen Diagramme aus CSVs neu erzeugt; schematische MLM-Abbildung fachlich korrigiert. Wirtschaftliche Hochrechnungen aus ungerundeten Ausgangswerten berechnet.
- 51 Literaturquellen / 158 Belegstellen geprüft; falsche ENISA-Berichte ersetzt, Dalton-Version aktualisiert, zahlreiche Fundstellen berichtigt. NIST SP 800-61r3 für den Prozessentwurf im Original geprüft. Aussagen zu drei nur im Abstract zugänglichen Arbeiten eingeschränkt; unbelegte URL-Vollständigkeitsquote entfernt.
- Betriebskonzept neu aufgebaut: unterstützende Webseitenvorprüfung, menschliche Fallentscheidung, begründete Wahl zwischen URL-Pfad, TRI Full und Kaskade, Datenstrategie sowie messbare Pilotkriterien. 63 weitere Rohscore-/Szenarioprüfungen; operative Precision von P≥90 und Modellrouting von menschlicher Prüfung getrennt. Lokalen Betriebsrechner mit ausdrücklich angenommenen Volumina und Prävalenzen ergänzt. Bewertung am tatsächlichen Bachelor-Maßstab neu begründet (90,5 Punkte / 1,5).
- Theorie und redundante Passagen gestrafft; zehn längere Codeauszüge in den Anhang verschoben. Umbrüche, Literaturabsätze, Abkürzungsverzeichnis, Tabellen, Captions und Diagramme visuell bereinigt.
- Deutsche und englische Zusammenfassung, KI-Erklärung samt tatsächlichen Produkten/Einsatzbereichen, lokale Kennzeichnungen und ehrenwörtliche Erklärung ergänzt. Keine Unterschriften oder unbekannten Modellversionen erzeugt.
- 120 zusätzliche CPU-Probes, Rohscore-/Bootstrap-Prüfung, Hash- und Seedregister sowie nachvollziehbare Prüfskripte und Reproduktionsarchive bereitgestellt. Kleine Neufit-Abweichungen und nicht rekonstruierbare Rohzeitmessungen transparent ausgewiesen.

Die Originaldateien wurden nicht überschrieben. Maschinenlesbare Änderungsprotokolle und historische Einzelbefunde liegen im Reproduktionspaket unter `review/`; entscheidend für den Schlussstand sind `REVIEW_FINAL.md`, `KERNPRUEFUNG.md` und `QUELLENPRUEFUNG.md`.
