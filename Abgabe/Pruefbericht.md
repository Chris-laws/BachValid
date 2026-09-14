# Unabhängiger Abschlussreview

Stand: 14.09.2026. Bewertet wird die überarbeitete Fassung von Chris David Kaufmann, „Self-Supervised Learning zur Phishing-Detektion – Konzeption, prototypische Entwicklung und Evaluation eines lernbasierten Klassifikationssystems“.

**Entscheidung: GERINGFÜGIGE RESTPUNKTE.** Die überarbeitete Arbeit ist kompiliert und inhaltlich für die Einreichung vorbereitet. Es fehlen die persönlichen Datierungen und Unterschriften auf zwei KI-Erklärungsseiten und der ehrenwörtlichen Erklärung. Diese persönlichen Erklärungen kann der Review nicht für den Verfasser abgeben. Die unten beschriebenen wissenschaftlichen Grenzen wurden in die Arbeit aufgenommen; sie erfordern keine erfundenen Daten oder nachträglich behauptete Vorregistrierung.

**Reviewbewertung als duale Bachelorarbeit: 90,5/100 Punkte, Note 1,5 („sehr gut“).** Das ist eine begründete Zweitgutachter-Einschätzung, keine amtliche Prüfungsnote und keine Garantie für die tatsächliche Bewertung. Die frühere Einschätzung von 87 Punkten wird durch diese Beurteilung ersetzt: Maßstab ist die im Formular verlangte praxisbezogene wissenschaftliche Problembearbeitung auf Bachelor-Niveau. Zusätzliche Grundlagenforschung, wiederholtes vollständiges Vortraining oder ein produktiver Bankbetrieb sind keine pauschalen Pflichtvoraussetzungen für eine sehr gute Bachelorarbeit. Das ergänzte Einsatzkonzept wird als überprüfbarer eigener Entwurf bewertet; es wird nicht als durchgeführter Pilot angerechnet.

## Tatsächliches DHBW-Bewertungsschema

Grundlage ist das vorgelegte `Gutachtenformular_PA-BA_09.01.2023.dotx`, nicht ein frei erfundenes Raster. Die enthaltene Notenskala ordnet 90,5–91,5 Punkten die Note 1,5 zu; eine 1,0 beginnt bei 98,5 Punkten. Das Formular verlangt ausdrücklich ein dem Studienjahr angemessenes Bewertungsniveau.

| Bewertungsschwerpunkt des Formulars | Maximum | Punkte | Begründung und Abzüge |
|---|---:|---:|---|
| Themenerfassung und Strukturierung | 20 | 19,0 | Relevante Problemstellung, klare Forschungsfragen und nachvollziehbare Abfolge von Repräsentation, Labelbudget und System. Das Einsatzkonzept verbindet die Befunde mit konkreten Bearbeitungs- und Ressourcenentscheidungen. Restabzug für den breiten technischen Umfang und die nur konzeptionell belegte betriebliche Ausgangslage. |
| Themenbearbeitung | 40 | 36,0 | Für eine Bachelorarbeit anspruchsvolle Umsetzung, umfangreiche unabhängige Nachrechnung, kritische Reflexion und ein aus tatsächlichen Betriebsraten abgeleiteter Prozess-/Pilotentwurf. Abzüge betreffen die begrenzte Bestätigungsstärke nachträglicher Tests, die fehlende externe Validierung des Gesamtsystems und die ungemessene betriebliche Verarbeitungskette. Die angemessen offengelegten Grenzen führen nicht zu einem zusätzlichen pauschalen Abzug für fehlende Forschungsprojekte außerhalb des Umfangs. |
| Quellenauswahl und Quellenauswertung | 30 | 26,5 | 51 einschlägige wissenschaftliche und praktische Quellen mit 158 Belegstellen; Aussagebezüge und Fundstellen korrigiert. NIST stützt die Prozesseinordnung, während der konkrete Entwurf als eigene Übertragung ausgewiesen ist. Restabzug für drei nur anhand des Originalabstracts prüfbare Bezüge und fehlende erhobene Unternehmensinformationen. |
| Formale Aspekte | 10 | 9,0 | Vollständige Verzeichnisse, DE/EN-Abstracts, konsistenter Quellenstil, KI-Erklärungen, kompilierbares Projekt und regelkonformer Umfang. Restabzüge: persönliche Erklärungen noch unsigniert; teils dichte technische Tabellen und lange Literaturangaben bleiben trotz Layoutbereinigung. |
| **Summe** | **100** | **90,5** | **Note 1,5 gemäß beigefügter Punkteskala** |

Die Untermerkmale des Originalformulars wurden vollständig berücksichtigt: Problem/Ziel, Themenerfassung, Gliederung und Aktualität; Terminologie, Methodenauswahl, Durchführung, kritische Analyse, eigene Ansätze und Reflexion; wissenschaftliche sowie praktische Quellen und kritische Distanz; äußere Form, sämtliche Verzeichnisse, Sprache, Umfang und Zitiertechnik. Es wurden keine amtlichen Teilgewichte für einzelne Untermerkmale erfunden.

Eine 1,0 ist nach diesem Dateienreview nicht hinreichend begründet: Vor allem die unternehmensspezifische Ausgangslage bleibt ungemessen, einzelne Bestätigungsansprüche sind methodisch begrenzt und drei Literaturbezüge sind nur anhand des Originalabstracts geprüft. Die Eigenständigkeit im prüfungsrechtlichen Sinn und die Fähigkeit, sämtliche Entscheidungen fachlich zu vertreten, müssen durch den Verfasser und die Hochschule beurteilt werden; ein Dateienreview kann dies nicht abschließend feststellen.

## Rekonstruiertes Untersuchungsdesign

Der eingefrorene PhreshPhish-Bestand umfasst SUP 4.000, DEV 20.000, SSL 200.000, CAL 50.000 ausschließlich benigne und FINAL 168.060 Instanzen. CAL bestimmt den angestrebten 0,5-%-FPR-Betriebspunkt; die tatsächlich realisierte FINAL-FPR kann davon abweichen. DEV-Auswertungen verwenden interne, vom jeweiligen Versuch abhängige Teilungen. Der Data Freeze umfasst 442.060 Rolleninstanzen.

R0 kombiniert allgemein vortrainiertes BERT für URLs und RoBERTa für HTML-Text. RT/RU/RUT passen Text, URL beziehungsweise beide Zweige zusätzlich per MLM an. Die linearen/MLP-Probes arbeiten auf eingefrorenen, zusammen 1.536-dimensionalen Mean-Pooling-Repräsentationen. Das spätere multimodale System trainiert Teile der Transformer weiter und ergänzt einen DOM-GCN sowie Gating. Die kontrastiven Entwicklungsversuche bilden einen gesonderten historischen Versuchspfad.

FF1 und die ursprünglichen FF2-Kurven verwenden N5. FF3 ergänzt fünf Downstream-Replikationen, ohne TAPT neu zu trainieren. XGBoost ist eine nachträgliche, unabhängig trainierte Referenz; seine Vergleiche sind ungepaart. Die nachträgliche sequenzielle FF4-Ablation N10 verwendet DEV und unterscheidet sich sowohl von der eingefrorenen FINAL-Kaskade als auch vom zusätzlichen N5-Frontendvergleich.

Putra ist eine externe Datenquelle mit internem frühen/späten Schnitt, aber kein zeitlich zukünftiger Test gegenüber PhreshPhish 2025. Quell-RUT gegen R0 und zusätzliche Putra-Anpassung gegen Quell-RUT sind zwei unterschiedliche Interventionen. Diese Trennungen sind jetzt in Methodik, Ergebnissen, Diskussion und Fazit konsistent.

## Prüfungen und Befunde

- Das ergänzte Betriebskonzept enthält einen Prozessentwurf, drei gemessene Betriebsalternativen und überprüfbare Pilotkriterien. 63 neue Prüfungen bestätigen die operativen Fallzahlen und die klassenspezifische Kaskadeneskalation aus FINAL-Rohscores/-Labels. Bei angenommenen 1 % Phishing und 10.000 Seiten ergeben sich für TRI Full 96,12 TP, 65,96 FP, 3,88 FN und 59,30 % Precision. Die höheren P≥90-Werte beziehen sich auf eine andere Schwelle. Die Kaskade benötigt bei reinem Prior-Shift auf 1 % rechnerisch 6,596 % zusätzliche Vollpfade; 14,933 % gilt für die ursprüngliche Testmischung. Diese Größen sind keine menschlichen Eskalationsraten. Der lokale Betriebsrechner wurde im Browser einschließlich Grenzfällen geprüft.
- 364 Prüfungen der Kennzahlen, Zählungen, Intervalle, Paarungen, Auswahlhashes und Statistik bestanden. Sämtliche FF1-Kontraste und Interaktionen, N10-Effekte, tatsächliche Holm-Familien, alle FF3-Budget-/Margenkombinationen, B95 und beide Putra-Versuche wurden unabhängig gerechnet.
- Die sechs FINAL-Scorevarianten wurden gegen wiedergewonnene Instanzlabels geprüft. Zusammen mit den 1.000 stratifizierten Bootstrap-Replikationen je Full/Kaskade stimmen 64 direkte Rohscore-/Bootstrap-Vergleiche überein. Der Float32-Grenzfall beim Routing wurde ausdrücklich reproduziert.
- 120 neue Putra-Linear-Probes aus gespeicherten Embeddings und identischen Auswahlhashes bestätigen Richtung und Sign-Flip-Entscheidung beider AP-AULC-Hauptbefunde. Von 480 Einzelmetrikvergleichen weichen 113 bei 10⁻⁸ Toleranz ab. Die größte AP-Abweichung beträgt 0,005491 Prozentpunkte. Die historischen Daten wurden nicht durch neue Solverergebnisse überschrieben.
- Für CAL, DEV, SSL und FINAL wurden Labelzahlen, interne SHA-Duplikate und alle sechs paarweisen SHA-Schnittmengen aus Metadaten neu geprüft: keine exakten Überschneidungen. Zeitquantil und 46.784 Late-Fälle einschließlich gebundener Datumswerte wurden neu bestimmt. SUP- sowie genaue Domain-/Template-Aussagen bleiben auf die archivierten Audits beschränkt.
- 51 Quellen und 158 einzelne Belegstellen der Schlussfassung sind dokumentiert. Originalfundstellen und Seitenzahlen wurden geprüft; bei Liu (2021), Rashid (2024) und Wang (2023) ausschließlich der Originalabstract. Deren Aussagen wurden entsprechend eingeschränkt. Das Register unterscheidet echte Originaldateien von eigenen Web-Prüfnotizen.
- Alle empirischen Diagramme wurden aus Ergebnisdateien neu erzeugt. Zahlen und Formulierungen in Tabellen, Abbildungen, Interpretation und Fazit wurden abgeglichen. Die letzten vier Abweichungen der Base-Rate-Tabelle wurden auf 95,62/96,54 % bei 5 % bzw. 80,74/84,27 % bei 1 % korrigiert.

Die ausführlichen Werte stehen in [Kernpruefung.md](Kernpruefung.md). Rechen- und Quellenprüfregister, historische Inputs, wiedergewonnene Labels/Embeddings und ausführbarer CPU-Prüfcode liegen in den Begleitarchiven.

## DHBW-Richtlinie und finale PDF

Maßgeblich ist die im aktuellen Studiengangskatalog verfügbare Mosbacher WI-Richtlinie vom 11.02.2026. Der offizielle [DHBW-Dokumentenordner](https://app.box.com/s/mcqho6icl8) wurde mit der vorhandenen Datei abgeglichen; der Katalogstand ist im Reproduktionspaket archiviert. Die Richtlinie und das tatsächliche Bewertungsformular liegen auch im Quellenarchiv.

| Regelbereich | Umsetzung und Prüfung |
|---|---|
| Umfang, Richtlinie S. 91 / Bewertungsformular | **57 Haupttextseiten einschließlich Abbildungen und Tabellen**, innerhalb 40–60; 82 physische PDF-Seiten insgesamt. |
| Schrift und Satz, S. 92 | Haupttext Times New Roman 11 pt, eineinhalbzeilig, 2,5-cm-Ränder. Tabellen/Code getrennt formatiert. |
| Abstracts, S. 73 | Deutsche Zusammenfassung und englischer Abstract, jeweils eine Seite. |
| KI-Erklärung, S. 56–58 | Vorgesehene Erklärung mit gewählter KI-Nutzung, zehn Einsatzbereichen, Produktnamen/Links und Verantwortungsangabe. Umfassender Einsatz auf den Textseiten, Abbildungen und Codeartefakten gekennzeichnet. Frühere unbekannte Modellversionen nicht erfunden. |
| Ehrenwörtliche Erklärung, S. 63 | Vorgegebener Wortlaut übernommen; persönliche Unterschrift bleibt offen. |
| Nachweise, S. 110–111 | Einheitliche Kurzbelege mit Autor/Jahr und geprüfter Fundstelle; abweichende Seitenzählungen ausdrücklich bezeichnet. |
| Digitale Artefakte, S. 42 | PDF, native LaTeX-Dateien, relevante Notebooks, Ergebnisartefakte und Quellenarchiv bereitgestellt. |

Die Fassung wurde mit Tectonic 0.17 vollständig kompiliert. Alle Seiten wurden gerendert und visuell kontrolliert; nach den letzten Eingriffen wurden die betroffenen Umbrüche erneut geprüft. Keine ungelösten Verweise und keine über den Satzspiegel ragenden Textobjekte in der finalen Prüfung. Verzeichnisse, Captions, Diagramme und Listings sind enthalten. Die verbleibenden TeX-Meldungen betreffen lockeren Blocksatz und Windows-Systemschriften, nicht fehlende Inhalte. Ein Schriftwechsel auf einem anderen Rechner kann Umbrüche ändern; deshalb wird die geprüfte PDF mitgeliefert.

## Verbleibende Risiken

1. **Persönlicher Abschluss:** Zwei KI-Seiten und die ehrenwörtliche Erklärung müssen vom Verfasser inhaltlich verantwortet, datiert und unterschrieben werden. Die breite tatsächliche KI-Nutzung wurde offengelegt; eine vollständig ohne KI erzeugte Eigenleistung wird nicht behauptet. Eine konkrete Produktversion früherer Sitzungen ist nicht rekonstruierbar.
2. **Inferenz und Generalisierung:** Feste TAPT-Checkpoints, kleine Replikationszahlen, Annahmen der Sign-Flip-/t-/Welch-Verfahren, überlappende Szenarien und nachträgliche Analysen begrenzen Bestätigungsansprüche. Der externe Test ist retrospektiv und betrifft Repräsentationen, nicht das vollständige System.
3. **Reproduktion:** Kein erneutes GPU-Vortraining; nicht alle ursprünglichen Modellcheckpoints und Rohrollen liegen im Paket. Absolute Laufzeitquantile und Durchsatzstreuungen können mangels Einzelmessungen nicht unabhängig rekonstruiert werden. Putra-CPU-Neufits sind hinsichtlich einzelner Metriken nicht bitgenau.
4. **Quellen:** Drei Originalabstracts statt Volltextprüfung. Die unmittelbaren Aussagen sind darauf beschränkt; weitergehende Detailaussagen wurden entfernt. Das Archiv behauptet keinen Zugriff auf unzugängliche Volltexte.
5. **Betriebliche Übertragung:** Das ergänzte Einsatzkonzept und der Betriebsrechner sind Szenariowerkzeuge. Es gibt keine prospektive Bank-Pilotierung, keine gemessene Gesamtprozess-/Personalersparnis und keine Garantie konstanter Fehler- oder Routingraten innerhalb der Klassen. Die 50k-Suffizienz gilt für lineare Probes; die operativen Full-/Kaskadenwerte stammen aus 200k-Training. Angenommene Prävalenzänderungen werden für Precision und Routing konsistent umgerechnet; die Laufzeit wird dabei nicht neu geschätzt.

Diese Grenzen sind in der Schlussfassung offen benannt. Die wissenschaftlich behebbaren Text-, Zahlen-, Quellen- und Konsistenzfehler wurden korrigiert. Die Empfehlung zur Einreichung gilt für die persönlich verantwortete und unterschriebene Fassung.
