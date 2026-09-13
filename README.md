# BachValid

Überarbeitete Bachelorarbeit und unabhängige wissenschaftliche Validierung zur Phishing-Detektion mit Self-Supervised Learning.

Stand: 13.09.2026. **GERINGFÜGIGE RESTPUNKTE:** Die Fassung ist kompiliert und visuell geprüft; die persönlichen Datierungen und drei Unterschriften bleiben offen. Umfang: 79 physische PDF-Seiten, davon 54 Haupttextseiten. Die unabhängige Revieweinschätzung nach dem vorgelegten DHBW-Schema beträgt 87/100 Punkte (1,7); sie ist keine amtliche Prüfungsnote.

| Einstieg | Inhalt |
|---|---|
| [Finale PDF](Abgabe/Bachelorarbeit_Kaufmann.pdf) | Geprüfte Schlussfassung |
| [LaTeX-Projekt](revised/README.md) · [ZIP](Abgabe/Bachelorarbeit_LaTeX.zip) | Kompilierbare Quellen und Abbildungen |
| [Kernprüfung](Abgabe/Kernpruefung.md) | FF1-Differenzen/Interaktion, N10, CIs, Sign-Flip, Holm, Nichtunterlegenheit/Margensensitivität, Putra, FF4 und Base Rate |
| [Quellenprüfung](review/QUELLENPRUEFUNG.md) | 50 Quellen und 156 einzelne Belegstellen; drei Quellen nur anhand des Originalabstracts geprüft |
| [Prüfbericht](Abgabe/Pruefbericht.md) · [Änderungen](Abgabe/Aenderungsprotokoll.md) | Bewertung, Korrekturen und verbleibende Grenzen |
| [Reproduktion](Reproduktion/README.md) | Skripte, historische Ergebnisartefakte und 13 Notebookkopien |

Die finale PDF ist bytegleich zur lokal geprüften Fassung. Die Tabellen-, Archiv- und Dateiprüfung ist in `review/FINAL_DELIVERY_VALIDATION.json` dokumentiert. 364 Aggregat-/Statistikprüfungen und 64 direkte Rohscore-/Bootstrap-Vergleiche bestanden. 120 zusätzliche Putra-Fits bestätigen die zentralen AP-AULC-Entscheidungen mit kleinen numerischen Abweichungen einzelner Metriken.

Die vollständigen Rohdaten-/Embeddingarchive und heruntergeladenen Literaturvolltexte bleiben im lokalen Abgabepaket. Ihre Veröffentlichung ist nicht Bestandteil dieses Git-Pushs. Der veröffentlichte Aggregataudit ist mit den enthaltenen Inputs ausführbar; die zusätzliche Rohscore-Reproduktion benötigt das lokale `artifacts/raw_recovery/`. Original-URLs/DOI stehen im Literaturverzeichnis, Herkunft und Hashes in den Prüfregistern.

Für den LaTeX-Build sind Tectonic oder XeLaTeX sowie Times New Roman, Arial und Consolas erforderlich. Das Literaturverzeichnis ist manuell in `revised/bibliography.tex` geführt. Ein Schriftwechsel kann Seitenumbrüche verändern.

KI-Unterstützung durch ChatGPT, teilweise Claude und den abschließenden Codex-Review ist in der Arbeit und den zugehörigen Codeartefakten dokumentiert. Die wissenschaftlichen Grenzen umfassen insbesondere feste TAPT-Checkpoints, nachträgliche Analysen, eine retrospektive externe Evaluation und fehlende einzelne Laufzeitmessungen.
