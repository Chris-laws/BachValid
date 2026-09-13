# Abschließende Kernprüfung: Zitate und Rechnungen

Stand: 13.09.2026. Alle folgenden Ergebnisse wurden aus Ergebniszeilen beziehungsweise Rohscores berechnet. Prozentwerte und Prozentpunktdifferenzen sind ausdrücklich getrennt. Die CSV-Dateien im Reproduktionspaket enthalten ungerundete Werte.

364 Aggregat-/Statistikprüfungen bestanden. Zusätzlich stimmen 64 Prüfungen der sechs FINAL-Scorevarianten und der Bootstrap-Auswertung mit den Originalergebnissen überein. 120 neue Putra-CPU-Fits bestätigen die AP-AULC-Aussagen mit kleinen numerischen Abweichungen; sie sind keine bitgenaue Reproduktion aller Metriken. Einzelne Laufzeitmessungen und vollständige Encodertrainings konnten nicht neu ausgeführt werden.

## Zitate

50 Quellen und 156 einzelne Belegstellen der Schlussfassung wurden registriert. Der jeweilige Aussagebezug und die angegebenen Fundstellen wurden an Originalquellen geprüft. Bei Liu (2021), Rashid (2024) und Wang (2023) war nur der Originalabstract zugänglich; die Aussagen sind auf dessen Inhalt begrenzt und tragen keine erfundenen PDF-Seitenzahlen. Einzelbefunde und Originalfundstellen stehen im [Quellenprüfregister](../review/QUELLENPRUEFUNG.md) und im Quellenarchiv.

## FF1: sämtliche Differenzen und Interaktion

DEV, 20.000 Labels, fünf gepaarte Wiederholungen. TPR jeweils aus TP/(TP+FN) rekonstruiert. Die sechs einfachen Kontraste je Probe sind ergänzende deskriptive Prüfungen; sie bilden keine nachträglich behauptete präregistrierte Testfamilie.

| Probe | Kontrast | Δ TPR (pp) | 95-%-t-KI (pp) | Sign-Flip p |
|---|---|---|---|---|
| LINEAR | RT-R0 | 6,204 | [4,364; 8,044] | 0,0625 |
| LINEAR | RU-R0 | 7,596 | [5,052; 10,140] | 0,0625 |
| LINEAR | RUT-R0 | 8,696 | [5,922; 11,470] | 0,0625 |
| LINEAR | RU-RT | 1,392 | [-0,264; 3,048] | 0,125 |
| LINEAR | RUT-RT | 2,492 | [1,270; 3,714] | 0,0625 |
| LINEAR | RUT-RU | 1,100 | [-0,246; 2,446] | 0,125 |
| LINEAR | RUT-RT-RU+R0 | -5,104 | [-6,755; -3,453] | 0,0625 |
| MLP | RT-R0 | 4,240 | [2,601; 5,879] | 0,0625 |
| MLP | RU-R0 | 7,320 | [6,311; 8,329] | 0,0625 |
| MLP | RUT-R0 | 8,732 | [7,444; 10,020] | 0,0625 |
| MLP | RU-RT | 3,080 | [1,341; 4,819] | 0,0625 |
| MLP | RUT-RT | 4,492 | [2,706; 6,278] | 0,0625 |
| MLP | RUT-RU | 1,412 | [0,782; 2,042] | 0,0625 |
| MLP | RUT-RT-RU+R0 | -2,828 | [-4,443; -1,213] | 0,0625 |

Interaktion = (RUT−RT)−(RU−R0). Bei N5 beträgt der kleinste mögliche zweiseitige Sign-Flip-p-Wert 0,0625. Kein FF1-N5-Kontrast ist nach diesem Test bei α=0,05 signifikant. Ein t-Intervall kann dennoch Null ausschließen: parametrisches Intervall und diskreter Sign-Flip-Test sind unterschiedliche Verfahren.

## N10: Equal Budget auf FINAL

RUT−R0 bei jeweils 20.000 Labels. Die Intervalle der Differenz werden aus den zehn gepaarten Differenzen berechnet, nicht durch Subtraktion zweier Einzelintervalle. Alle fünf Kontraste haben zehn positive Paare.

| Bedingung | R0 TPR (%) | RUT TPR (%) | Δ (pp) | 95-%-KI Δ (pp) | p roh | p Holm |
|---|---|---|---|---|---|---|
| OFFICIAL_TEST | 71,4277 | 77,5115 | 6,0837 | [5,3810; 6,7864] | 0,001953125 | — |
| DOMAIN_OOD_EXACT | 72,5632 | 78,2072 | 5,6441 | [4,9479; 6,3402] | 0,001953125 | 0,0078125 |
| TEMPLATE_OOD_EXACT | 71,1251 | 77,3811 | 6,2560 | [5,4949; 7,0170] | 0,001953125 | 0,0078125 |
| DOMAIN_TEMPLATE_OOD_EXACT | 72,2566 | 78,1435 | 5,8870 | [5,1286; 6,6453] | 0,001953125 | 0,0078125 |
| LATE_TEST_Q4 | 68,5567 | 74,0620 | 5,5052 | [4,3376; 6,6729] | 0,001953125 | 0,0078125 |

Official ist der einzelne Hauptvergleich; Holm gilt separat für die vier ergänzenden Bedingungen. Die fünf Bedingungen überlappen. Sie sind keine unabhängigen Replikationen. N10 variiert Downstream-Auswahl und Training bei festen TAPT-Checkpoints.

Zusätzlich unabhängig berechnete Intervalle der Einzelmittel (keine simultanen Intervalle):

| Bedingung | Variante | TPR-Mittel (%) | 95-%-t-KI (%) |
|---|---|---|---|
| DOMAIN_OOD_EXACT | R0 | 72,5632 | [71,8812; 73,2451] |
| DOMAIN_TEMPLATE_OOD_EXACT | R0 | 72,2566 | [71,4971; 73,0161] |
| LATE_TEST_Q4 | R0 | 68,5567 | [67,4511; 69,6623] |
| OFFICIAL_TEST | R0 | 71,4277 | [70,7489; 72,1065] |
| TEMPLATE_OOD_EXACT | R0 | 71,1251 | [70,3711; 71,8791] |
| DOMAIN_OOD_EXACT | RUT | 78,2072 | [77,4666; 78,9479] |
| DOMAIN_TEMPLATE_OOD_EXACT | RUT | 78,1435 | [77,4511; 78,8360] |
| LATE_TEST_Q4 | RUT | 74,0620 | [72,8134; 75,3106] |
| OFFICIAL_TEST | RUT | 77,5115 | [76,7844; 78,2385] |
| TEMPLATE_OOD_EXACT | RUT | 77,3811 | [76,7033; 78,0588] |

Alle weiteren vorhandenen N10-Budgetmittel und TPR-/FPR-Intervalle: `review/closure_n10_means_ci.csv` im Reproduktionspaket.

## FF3: Nichtunterlegenheit und Margensensitivität

Nullhypothese je Bedingung: ΔTPR ≤ −Marge. Gegen R0@200k wird der einseitige gepaarte Sign-Flip-Test auf d+Marge verwendet, gegen unabhängig trainiertes XGBoost@200k der einseitige Welch-Test. Für die gemeinsame Entscheidung gilt p_IUT = Maximum der fünf Bedingungs-p-Werte; anschließend Holm über sieben Budgets je Referenz und Marge. Die vier Margen sind eine ergänzende Sensitivitätsanalyse, keine zusätzliche gepoolte Bestätigungsfamilie.

Holm-p-Werte für alle Budgets:

| Referenz | Marge (pp) | 2000 | 5000 | 10000 | 20000 | 50000 | 100000 | 200000 |
|---|---|---|---|---|---|---|---|---|
| R0_200K | 0,0 | 1 | 1 | 1 | 1 | 0,0068359375 | 0,0068359375 | 0,0068359375 |
| R0_200K | 0,5 | 1 | 1 | 1 | 1 | 0,0068359375 | 0,0068359375 | 0,0068359375 |
| R0_200K | 1,0 | 1 | 1 | 1 | 1 | 0,0068359375 | 0,0068359375 | 0,0068359375 |
| R0_200K | 2,0 | 1 | 1 | 1 | 0,59375 | 0,0068359375 | 0,0068359375 | 0,0068359375 |
| XGB_200K | 0,0 | 1 | 1 | 1 | 1 | 6,285592018e-05 | 5,316965767e-10 | 1,069813229e-12 |
| XGB_200K | 0,5 | 1 | 1 | 1 | 0,6599103461 | 2,761939663e-05 | 2,812432284e-10 | 7,080219228e-13 |
| XGB_200K | 1,0 | 1 | 1 | 1 | 0,1732329335 | 1,280607815e-05 | 1,535429933e-10 | 4,771265694e-13 |
| XGB_200K | 2,0 | 1 | 1 | 1 | 0,008463645111 | 3,166332223e-06 | 4,975123327e-11 | 2,27405774e-13 |

Die kleinste gemeinsam gegenüber beiden Referenzen gestützte Budgetstufe beträgt bei 0 / 0,5 / 1 / 2 pp jeweils **50.000 Labels**. RUT20k besteht gegenüber XGBoost erst bei 2 pp; gegenüber R0 auch dann nicht. Bei 1 pp sind die fünf einzelnen XGBoost-Tests für 20k ausreichend, die Holm-korrigierte gemeinsame Budgetentscheidung jedoch nicht.

Effektgrößen bei 50k (Spanne der fünf Bedingungen; keine Konfidenzintervalle):

| Referenz | Δ TPR (pp) | Δ FPR (pp) | p IUT | p Holm |
|---|---|---|---|---|
| R0_200K | [3,2375; 3,6250] | [-0,1804; -0,0790] | 0,0009765625 | 0,0068359375 |
| XGB_200K | [4,5934; 5,5842] | [-0,3981; -0,0713] | 2,561215629e-06 | 1,280607815e-05 |

50.000 statt 200.000 bedeutet 75 % weniger gelabelte Downstream-Trainingsinstanzen. Es ist keine Messung einer 75-%-Einsparung sämtlicher Annotationskosten: DEV/CAL/Testlabels und die labelbasierte balancierte Auswahl gehören nicht zu diesem Budget. Der SSL-Pool bleibt 200.000 groß; sein Vollbudget ist mit 101.578 benignen und 98.422 Phishing-Fällen nicht exakt balanciert.

B95: 95 % des jeweiligen eigenen 200k-Endniveaus; DEV und FINAL getrennt:

| Bestand | Probe | Variante | B95 |
|---|---|---|---|
| DEV | LINEAR | R0 | 50000 |
| DEV | LINEAR | RUT | 10000 |
| DEV | MLP | R0 | 100000 |
| DEV | MLP | RUT | 50000 |
| FINAL | LINEAR | R0 | 50000 |
| FINAL | LINEAR | RUT | 50000 |
| FINAL | MLP | R0 | 100000 |
| FINAL | MLP | RUT | 100000 |

## Putra: AP-AULC und alle vier Budgetdifferenzen

AP-AULC wird je Lauf als trapezförmiges Integral der AP über log(Budget), geteilt durch log(2000)−log(200), berechnet. Erst danach werden die zehn Flächen gepaart verglichen. Logarithmusbasis ändert die normalisierte Fläche nicht. Die 2.598 späteren Putra-Fälle bilden den festen Test; der frühere Pool enthält 7.793 Fälle. Der gesamte Putra-Bestand liegt zeitlich vor PhreshPhish 2025.

### Quell-RUT gegenüber R0

AP-AULC: 89,4485 % → 91,9836 %. Δ = **2,5351 pp**, 95-%-KI [2,1609; 2,9092] pp; p = 0,001953125.

| Labels | AP Basis (%) | AP angepasst (%) | Δ AP (pp) | 95-%-KI (pp) | positive/negative Paare | p roh | p Holm |
|---|---|---|---|---|---|---|---|
| 200 | 83,9117 | 86,5146 | 2,6029 | [1,7003; 3,5055] | 10/0 | 0,001953125 | 0,0078125 |
| 500 | 88,9624 | 91,7567 | 2,7944 | [2,2246; 3,3641] | 10/0 | 0,001953125 | 0,0078125 |
| 1000 | 91,6965 | 94,0495 | 2,3530 | [2,0208; 2,6853] | 10/0 | 0,001953125 | 0,0078125 |
| 2000 | 93,4008 | 95,6083 | 2,2075 | [1,8516; 2,5633] | 10/0 | 0,001953125 | 0,0078125 |

### Zusätzliche Zielanpassung E1 gegenüber Quell-RUT E0

AP-AULC: 91,9836 % → 90,8447 %. Δ = **-1,1389 pp**, 95-%-KI [-1,4994; -0,7785] pp; p = 0,001953125.

| Labels | AP Basis (%) | AP angepasst (%) | Δ AP (pp) | 95-%-KI (pp) | positive/negative Paare | p roh | p Holm |
|---|---|---|---|---|---|---|---|
| 200 | 86,5146 | 84,2341 | -2,2805 | [-3,3763; -1,1847] | 1/9 | 0,00390625 | 0,01171875 |
| 500 | 91,7567 | 90,4421 | -1,3146 | [-1,8969; -0,7323] | 0/10 | 0,001953125 | 0,0078125 |
| 1000 | 94,0495 | 93,5547 | -0,4948 | [-0,7514; -0,2383] | 1/9 | 0,005859375 | 0,01171875 |
| 2000 | 95,6083 | 95,0981 | -0,5102 | [-0,7506; -0,2697] | 1/9 | 0,005859375 | 0,01171875 |

Die beiden vierteiligen Budgetfamilien werden separat korrigiert. Die AP-AULC-Hauptvergleiche sind keine fünften Budgettests. Der positive Transferbefund und die negative weitere Zielanpassung betreffen unterschiedliche Interventionen.

120 neue Logistic-Regression-Fits aus wiedergewonnenen Embeddings und identischen Auswahlhashes ergeben ΔAP-AULC +2,535073 pp bzw. −1,138777 pp; beide p=0,001953125. Die historischen Werte bleiben als ursprüngliche Versuchsergebnisse erhalten. Von 480 Metrikvergleichen weichen 113 bei einer Toleranz von 10⁻⁸ ab. Größte Abweichungen: AP 0,005491 pp, AUC 0,001691 pp, FPR@TPR90 0,059312 pp, P≥90 0,086337 pp. Ursachen können Solver-/Umgebungsunterschiede sein; eine konkrete Ursache wurde nicht isoliert.

## FF4: Komponenten, finale Abstände, Laufzeit und Eskalation

Sequenzielle DEV-Komponentenablation, N10, gleiche Labels und gepaarte Seeds:

| Variante | TPR (%) | FPR (%) | FPR@TPR90 (%) |
|---|---|---|---|
| P0_URL | 87,932 | 0,400 | 0,528 |
| P1_DUAL_EQUAL | 93,116 | 0,488 | 0,316 |
| P2_TRI_EQUAL | 93,844 | 0,568 | 0,296 |
| P3_TRI_GATED | 94,420 | 0,616 | 0,260 |

| Kontrast | Δ TPR (pp) | 95-%-KI (pp) | p roh | p Holm |
|---|---|---|---|---|
| P1_DUAL_EQUAL-P0_URL | 5,184 | [4,324; 6,044] | 0,001953125 | 0,005859375 |
| P2_TRI_EQUAL-P1_DUAL_EQUAL | 0,728 | [0,305; 1,151] | 0,001953125 | 0,005859375 |
| P3_TRI_GATED-P2_TRI_EQUAL | 0,576 | [0,226; 0,926] | 0,00390625 | 0,005859375 |

Beim Gating sind neun Differenzen positiv und eine null; deshalb p roh=0,00390625. Holm ist für alle drei Kontraste 0,005859375. Diese nachträgliche Komponentenanalyse ist von der eingefrorenen FINAL-Systemauswertung zu unterscheiden.

Finale Systemabstände aus TP/FP-Zählungen; deskriptive Einzelmodellvergleiche:

| Bedingung | Kontrast | Δ TPR (pp) | Δ FPR (pp) |
|---|---|---|---|
| DOMAIN_OOD_EXACT | gated-equal | 0,3960 | -0,0632 |
| DOMAIN_OOD_EXACT | cascade-gated | -0,2181 | 0,0194 |
| DOMAIN_TEMPLATE_OOD_EXACT | gated-equal | 0,3787 | -0,0635 |
| DOMAIN_TEMPLATE_OOD_EXACT | cascade-gated | -0,2082 | 0,0196 |
| LATE_TEST_Q4 | gated-equal | 0,5410 | -0,0953 |
| LATE_TEST_Q4 | cascade-gated | -0,1527 | 0,0106 |
| OFFICIAL_TEST | gated-equal | 0,4635 | -0,0405 |
| OFFICIAL_TEST | cascade-gated | -0,3490 | 0,0142 |
| TEMPLATE_OOD_EXACT | gated-equal | 0,4498 | -0,0410 |
| TEMPLATE_OOD_EXACT | cascade-gated | -0,3384 | 0,0144 |

| Modus | TP | FP | TPR (%) | FPR (%) | AP (%) | FPR@TPR90 (%) |
|---|---|---|---|---|---|---|
| url | 71544 | 622 | 93,1562 | 0,6816 | 99,3680 | 0,3364 |
| text | 34014 | 597 | 44,2891 | 0,6542 | 94,5954 | 12,7427 |
| gated | 73819 | 608 | 96,1185 | 0,6662 | 99,6791 | 0,2170 |
| equal | 73463 | 645 | 95,6549 | 0,7068 | 99,5997 | 0,4032 |
| dom | 19553 | 388 | 25,4596 | 0,4252 | 91,0667 | 33,5163 |
| cascade | 73551 | 621 | 95,7695 | 0,6805 | 98,5385 | 0,1698 |

Rohscore-Reproduktion der Rankingmetriken mit je 1.000 stratifizierten Bootstrap-Stichproben:

| System | AP (%) | 95-%-Percentile-KI AP (%) | P≥90 (%) | 95-%-Percentile-KI P≥90 (%) |
|---|---|---|---|---|
| FULL | 99,679116 | [99,659258; 99,699499] | 99,714422 | [99,673365; 99,753322] |
| CASCADE | 98,538510 | [98,476036; 98,601942] | 99,776393 | [99,741715; 99,815322] |

DEV-In-Memory-Benchmark: Median 30,182580 → 11,953473 ms; Quotient Full/Kaskade **2,525005**, entsprechend **60,396119 %** geringerer Median-Latenz. Durchsatz 183,176483 → 447,019312 Seiten/s; Faktor **2,440375** bzw. 144,0375 % Zunahme.

95. Latenzperzentil: 47,402349 → 32,136478 ms; Durchsatz-SD über drei Läufe: 1,32461 → 2,95146 Seiten/s. Die absoluten Zeitquantile und Streuungen sind gegen Benchmark-Code und gespeicherte Aggregate geprüft. Rohzeitreihen fehlen; diese Werte wurden nicht unabhängig neu gemessen. Keine Netzwerk-, Rendering-, Datenträger- oder vorgelagerten HTML-Parsingkosten enthalten.

Alle vier gespeicherten Benchmarkpfade; Quotienten aus ungerundeten Aggregaten:

| System | Median (ms) | p95 (ms) | Seiten/s | Median-Faktor Full/Variante | p95-Faktor Full/Variante | Durchsatz-Faktor Variante/Full |
|---|---|---|---|---|---|---|
| URL_ONLY_FROM_TRI_RUT | 11,1900 | 12,3285 | 865,8598 | 2,697272 | 3,844946 | 4,726916 |
| TRI_RUT_200K_FULL | 30,1826 | 47,4023 | 183,1765 | 1,000000 | 1,000000 | 1,000000 |
| TRI_RUT_200K_CASCADE | 11,9535 | 32,1365 | 447,0193 | 2,525005 | 1,475033 | 2,440375 |
| DUAL_RUT_200K_FULL | 26,0063 | 40,5398 | 203,7561 | 1,160586 | 1,169279 | 1,112349 |

Eingefrorene FINAL-Kaskade: 25097 / 168060 = **14,933357 % Eskalation**, 85,066643 % frühe Entscheidungen. Historische Float32-Grenzvergleiche wurden explizit reproduziert; eine veränderte Float64-Grenzbehandlung würde 15,050577 % erzeugen und wäre ein anderes Routing.

Separater N5-Frontendvergleich auf Official, alle vier DEV-Toleranzen:

| Toleranz (pp) | Frontend | Eskalation (%) | frühe Entscheidung (%) | Δ TPR zu Full (pp) | Δ FPR zu Full (pp) |
|---|---|---|---|---|---|
| 0,25 | URL_BASE | 48,2192 | 51,7808 | -0,0750 | 0,0552 |
| 0,25 | URL_TAPT | 28,4983 | 71,5017 | -0,4201 | 0,0399 |
| 0,50 | URL_BASE | 40,0624 | 59,9376 | -0,3448 | 0,0414 |
| 0,50 | URL_TAPT | 27,2920 | 72,7080 | -0,4948 | 0,0344 |
| 1,00 | URL_BASE | 35,7460 | 64,2540 | -0,6865 | 0,0265 |
| 1,00 | URL_TAPT | 23,5139 | 76,4861 | -0,9401 | 0,0160 |
| 2,00 | URL_BASE | 28,3974 | 71,6026 | -1,2539 | 0,0136 |
| 2,00 | URL_TAPT | 21,8105 | 78,1895 | -1,2948 | -0,0046 |

Bei 1 pp sinkt Official-Eskalation von 35,746043 % auf 23,513864 % (−12,232179 pp). Der ungewichtete Mittelwert über fünf überlappende Szenarien beträgt dagegen 41,124611 % → 28,727791 % (−12,396819 pp). Er ist keine Populationsrate. Beide Werte gehören zu einem anderen Routingprotokoll als die eingefrorenen 14,933357 %.

## Base-Rate-Umrechnung

P≥90 bezeichnet die höchste empirische Precision unter Schwellen mit Recall ≥0,9. FPR@TPR90 verwendet dagegen den ersten ROC-Punkt mit TPR ≥0,9. Beide Größen dürfen nicht ohne Schwellenabgleich kombiniert werden.

Mit π₀=76.800/168.060=0,4569796501249554 und der ungerundeten P≥90-Precision gilt λ=((1−P₀)/P₀)·π₀/(1−π₀)=FPR*/TPR*. Daraus folgt Precision(π)=π/[π+(1−π)λ]. Die unabhängige äquivalente Bayes-Rechnung liefert dieselben Werte.

| System | Prävalenz (%) | Precision (%) | Schlussfassung gerundet (%) |
|---|---|---|---|
| FULL | 45,697965 | 99,714422 | 99,714 |
| FULL | 5,000000 | 95,621194 | 95,62 |
| FULL | 1,000000 | 80,735859 | 80,74 |
| CASCADE | 45,697965 | 99,776393 | 99,776 |
| CASCADE | 5,000000 | 96,540589 | 96,54 |
| CASCADE | 1,000000 | 84,266395 | 84,27 |

Korrektur der letzten noch abweichenden Text-/Tabellenwerte: bei 5 % 95,61→95,62 % und 96,53→96,54 %; bei 1 % 80,71→80,74 % und 84,24→84,27 %. Die Umrechnung gilt bei konstantem klassenspezifischem Scoreverhalten und ist kein externer empirischer Benchmark.

## Reproduktionsweg

`Reproduktion.zip` entpacken; dort `README.md` folgen. Hauptskripte: `empirical_audit.py`, `raw_score_audit.py`, `closure_audit.py`. Dieses Protokoll wird mit `core_report.py` aus den geprüften CSVs erzeugt. Sign-Flip-p-Werte beruhen auf vollständiger Vorzeichenenumeration, setzen für die inferenzielle Interpretation jedoch Symmetrie/Austauschbarkeit voraus. t-Intervalle beschreiben die Variation der gegebenen Replikationen; sie ersetzen keine unabhängigen Daten- oder Vortrainingswiederholungen.
