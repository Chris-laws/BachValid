# Betriebskonzept: Rechen- und Herkunftsprüfung

Die Szenarien sind eine nachträgliche Interpretation der eingefrorenen Ergebnisse. Sie sind keine Messungen bei der Volksbank und kein neuer Systemtest.

Prüfmodus: raw FINAL labels and scores; all three confusion matrices checked. 63 Prüfungen bestanden.

## Operative Fallzahlen

Je 10.000 klassifizierbare Webseiten; gleiche klassenspezifische Fehler wie auf OFFICIAL_TEST. Alle drei Betriebsarten stammen aus TRI_RUT_200000, nicht aus FF3-RUT50k.

| Phishinganteil | Modell | TP | FP | FN | Modellpositiv | Precision | zusätzliche Vollpfade |
|---|---|---:|---:|---:|---:|---:|---:|
| 0,10 % | CASCADE | 9,58 | 67,98 | 0,42 | 77,56 | 12,35 % | 642,78 |
| 0,10 % | FULL | 9,61 | 66,56 | 0,39 | 76,17 | 12,62 % | 10000,00 |
| 0,10 % | URL | 9,32 | 68,09 | 0,68 | 77,40 | 12,04 % | 0,00 |
| 1,00 % | CASCADE | 95,77 | 67,37 | 4,23 | 163,14 | 58,71 % | 659,57 |
| 1,00 % | FULL | 96,12 | 65,96 | 3,88 | 162,08 | 59,30 % | 10000,00 |
| 1,00 % | URL | 93,16 | 67,48 | 6,84 | 160,63 | 57,99 % | 0,00 |
| 5,00 % | CASCADE | 478,85 | 64,64 | 21,15 | 543,49 | 88,11 % | 734,18 |
| 5,00 % | FULL | 480,59 | 63,29 | 19,41 | 543,88 | 88,36 % | 10000,00 |
| 5,00 % | URL | 465,78 | 64,75 | 34,22 | 530,53 | 87,80 % | 0,00 |
| 45,70 % | CASCADE | 4376,47 | 36,95 | 193,32 | 4413,42 | 99,16 % | 1493,34 |
| 45,70 % | FULL | 4392,42 | 36,18 | 177,38 | 4428,60 | 99,18 % | 10000,00 |
| 45,70 % | URL | 4257,05 | 37,01 | 312,75 | 4294,06 | 99,14 % | 0,00 |

## Entscheidungsrelevante Trennungen

- Die Precision hier gilt an der auf CAL bestimmten Schwelle. Die vorhandene P≥90-Prävalenzrechnung benutzt eine andere empirische Schwelle. Ihre 80,74/84,27 % bei 1 % dürfen nicht mit den hiesigen Betriebsraten kombiniert werden.
- Kaskadenrouting aus Rohscores: 5849/91260 benign (6,409 %) und 19248/76800 Phishing (25,062 %). Bei verändertem Prior gilt e(π)=(1−π)e0+πe1. 14,933 % gilt nur für die ursprüngliche Mischung.
- Zusätzlicher Vollpfad, Modellalarm und menschlicher Prüffall sind verschiedene Größen. Bei Prüfung aller Modellpositiven ist TP+FP nur der entsprechende Teil der Prüfmenge; Negativstichproben, nicht auswertbare Fälle und weiterer Kontext kommen hinzu.
- In-Memory-Referenzzeit = 10.000 / gemessener mittlerer Durchsatz. Keine End-to-End-Prognose und keine Laufzeitprognose für eine geänderte Prävalenz. Einzelzeitreihen und klassenspezifische Zeitmessungen fehlen.
- 50k/75 % ist der Nachweis für das interne Linear-Probe-Training. Die hier betrachteten Systemmodelle nutzten 200k Trainingslabels. Kalibrations-, Entwicklungs- und Evaluationslabels sowie Erhebung und TAPT sind getrennte Ressourcen.
- Der vorgeschlagene Einsatzprozess und seine Freigabekriterien sind eigene Konzeption. Die Umsetzung von Erfassung, Tickets, Menschprüfung und Monitoring wurde nicht experimentell validiert.

## Quellenbezug

NIST SP 800-61r3 (Nelson, Rekhi, Souppaya, Scarfone; April 2025), gedruckte S. 25: technische Filterung für menschliche Analyse; S. 26–27: kontextbezogene Priorisierung und Validierung von Meldungen. Original geprüft: https://doi.org/10.6028/NIST.SP.800-61r3. Daraus folgt keine NIST-Empfehlung für dieses konkrete Modell.

## Reproduktion

`python scripts/business_analysis.py --require-raw` prüft die Rohscores/Labels erneut. Ohne diesen Parameter und ohne Rohlabels verwendet das öffentliche Paket `review/business_routing_counts.csv`; dieser Modus ist ausdrücklich im Ergebnis ausgewiesen. Alle Eingabedateien werden gehasht. Die Berechnung verändert weder Modelle noch historische Ergebnisartefakte.
