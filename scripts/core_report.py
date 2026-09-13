"""KI-unterstützte Ausgabe des Kernprüfprotokolls aus unabhängig berechneten CSVs."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'review';O=ROOT/'Abgabe';O.mkdir(exist_ok=True)
def read(name):return pd.read_csv(R/name)
def f(x,n=4):return f'{x:.{n}f}'.replace('.',',')
def p(x):return '—' if pd.isna(x) else f'{x:.10g}'.replace('.',',')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
S=['# Abschließende Kernprüfung: Zitate und Rechnungen', '', 'Stand: 13.09.2026. Alle folgenden Ergebnisse wurden aus Ergebniszeilen beziehungsweise Rohscores berechnet. Prozentwerte und Prozentpunktdifferenzen sind ausdrücklich getrennt. Die CSV-Dateien im Reproduktionspaket enthalten ungerundete Werte.', '',
'364 Aggregat-/Statistikprüfungen bestanden. Zusätzlich stimmen 64 Prüfungen der sechs FINAL-Scorevarianten und der Bootstrap-Auswertung mit den Originalergebnissen überein. 120 neue Putra-CPU-Fits bestätigen die AP-AULC-Aussagen mit kleinen numerischen Abweichungen; sie sind keine bitgenaue Reproduktion aller Metriken. Einzelne Laufzeitmessungen und vollständige Encodertrainings konnten nicht neu ausgeführt werden.', '',
'## Zitate', '',
'50 Quellen und 156 einzelne Belegstellen der Schlussfassung wurden registriert. Der jeweilige Aussagebezug und die angegebenen Fundstellen wurden an Originalquellen geprüft. Bei Liu (2021), Rashid (2024) und Wang (2023) war nur der Originalabstract zugänglich; die Aussagen sind auf dessen Inhalt begrenzt und tragen keine erfundenen PDF-Seitenzahlen. Einzelbefunde und Originalfundstellen stehen im [Quellenprüfregister](../review/QUELLENPRUEFUNG.md) und im Quellenarchiv.', '',
'## FF1: sämtliche Differenzen und Interaktion', '',
'DEV, 20.000 Labels, fünf gepaarte Wiederholungen. TPR jeweils aus TP/(TP+FN) rekonstruiert. Die sechs einfachen Kontraste je Probe sind ergänzende deskriptive Prüfungen; sie bilden keine nachträglich behauptete präregistrierte Testfamilie.', '']
a=read('FF1_ALL_CONTRASTS.csv')
S += [table(['Probe','Kontrast','Δ TPR (pp)','95-%-t-KI (pp)','Sign-Flip p'],[[r.probe,r.contrast,f(r.mean_delta_pp,3),f'[{f(r.ci95_lo_pp,3)}; {f(r.ci95_hi_pp,3)}]',p(r.p_exact)] for r in a.itertuples()]),'',
'Interaktion = (RUT−RT)−(RU−R0). Bei N5 beträgt der kleinste mögliche zweiseitige Sign-Flip-p-Wert 0,0625. Kein FF1-N5-Kontrast ist nach diesem Test bei α=0,05 signifikant. Ein t-Intervall kann dennoch Null ausschließen: parametrisches Intervall und diskreter Sign-Flip-Test sind unterschiedliche Verfahren.', '',
'## N10: Equal Budget auf FINAL', '',
'RUT−R0 bei jeweils 20.000 Labels. Die Intervalle der Differenz werden aus den zehn gepaarten Differenzen berechnet, nicht durch Subtraktion zweier Einzelintervalle. Alle fünf Kontraste haben zehn positive Paare.', '']
a=read('empirical_ff3_equal_budget.csv')
order=['OFFICIAL_TEST','DOMAIN_OOD_EXACT','TEMPLATE_OOD_EXACT','DOMAIN_TEMPLATE_OOD_EXACT','LATE_TEST_Q4']
a=a.set_index('scenario').loc[order].reset_index()
S += [table(['Bedingung','R0 TPR (%)','RUT TPR (%)','Δ (pp)','95-%-KI Δ (pp)','p roh','p Holm'],[[r.scenario,f(100*r.R0_tpr),f(100*r.RUT_tpr),f(100*r.mean_delta),f'[{f(100*r.ci95_lo)}; {f(100*r.ci95_hi)}]',p(r.p_raw),p(r.p_holm_secondary)] for r in a.itertuples()]),'',
'Official ist der einzelne Hauptvergleich; Holm gilt separat für die vier ergänzenden Bedingungen. Die fünf Bedingungen überlappen. Sie sind keine unabhängigen Replikationen. N10 variiert Downstream-Auswahl und Training bei festen TAPT-Checkpoints.', '',
'Zusätzlich unabhängig berechnete Intervalle der Einzelmittel (keine simultanen Intervalle):','']
b=read('closure_n10_means_ci.csv');b=b[b.budget.eq(20000)&b.metric.eq('tpr')&b.rep.isin(['R0','RUT'])]
S += [table(['Bedingung','Variante','TPR-Mittel (%)','95-%-t-KI (%)'],[[r.scenario,r.rep,f(r.mean_pct),f'[{f(r.ci95_lo_pct)}; {f(r.ci95_hi_pct)}]'] for r in b.itertuples()]),'',
'Alle weiteren vorhandenen N10-Budgetmittel und TPR-/FPR-Intervalle: `review/closure_n10_means_ci.csv` im Reproduktionspaket.', '',
'## FF3: Nichtunterlegenheit und Margensensitivität', '',
'Nullhypothese je Bedingung: ΔTPR ≤ −Marge. Gegen R0@200k wird der einseitige gepaarte Sign-Flip-Test auf d+Marge verwendet, gegen unabhängig trainiertes XGBoost@200k der einseitige Welch-Test. Für die gemeinsame Entscheidung gilt p_IUT = Maximum der fünf Bedingungs-p-Werte; anschließend Holm über sieben Budgets je Referenz und Marge. Die vier Margen sind eine ergänzende Sensitivitätsanalyse, keine zusätzliche gepoolte Bestätigungsfamilie.', '',
'Holm-p-Werte für alle Budgets:', '']
a=read('empirical_ff3_ni_global.csv')
budgets=[2000,5000,10000,20000,50000,100000,200000]
rows=[]
for (ref,margin),q in a.groupby(['reference','margin']):
    q=q.set_index('budget');rows.append([ref,f(100*margin,1)]+[p(q.loc[v,'p_holm']) for v in budgets])
S += [table(['Referenz','Marge (pp)']+[str(x) for x in budgets],rows),'',
'Die kleinste gemeinsam gegenüber beiden Referenzen gestützte Budgetstufe beträgt bei 0 / 0,5 / 1 / 2 pp jeweils **50.000 Labels**. RUT20k besteht gegenüber XGBoost erst bei 2 pp; gegenüber R0 auch dann nicht. Bei 1 pp sind die fünf einzelnen XGBoost-Tests für 20k ausreichend, die Holm-korrigierte gemeinsame Budgetentscheidung jedoch nicht.', '',
'Effektgrößen bei 50k (Spanne der fünf Bedingungen; keine Konfidenzintervalle):', '']
b=a[a.budget.eq(50000)&a.margin.eq(.01)]
S += [table(['Referenz','Δ TPR (pp)','Δ FPR (pp)','p IUT','p Holm'],[[r.reference,f'[{f(100*r.worst_mean_delta)}; {f(100*r.max_mean_delta)}]',f'[{f(100*r.min_delta_fpr)}; {f(100*r.max_delta_fpr)}]',p(r.p_iut),p(r.p_holm)] for r in b.itertuples()]),'',
'50.000 statt 200.000 bedeutet 75 % weniger gelabelte Downstream-Trainingsinstanzen. Es ist keine Messung einer 75-%-Einsparung sämtlicher Annotationskosten: DEV/CAL/Testlabels und die labelbasierte balancierte Auswahl gehören nicht zu diesem Budget. Der SSL-Pool bleibt 200.000 groß; sein Vollbudget ist mit 101.578 benignen und 98.422 Phishing-Fällen nicht exakt balanciert.', '',
'B95: 95 % des jeweiligen eigenen 200k-Endniveaus; DEV und FINAL getrennt:', '']
b=read('empirical_b95.csv');b=b[b.rep.isin(['R0','RUT'])]
S += [table(['Bestand','Probe','Variante','B95'],[[r.role,r.probe,r.rep,r.B95] for r in b.itertuples()]),'',
'## Putra: AP-AULC und alle vier Budgetdifferenzen', '',
'AP-AULC wird je Lauf als trapezförmiges Integral der AP über log(Budget), geteilt durch log(2000)−log(200), berechnet. Erst danach werden die zehn Flächen gepaart verglichen. Logarithmusbasis ändert die normalisierte Fläche nicht. Die 2.598 späteren Putra-Fälle bilden den festen Test; der frühere Pool enthält 7.793 Fälle. Der gesamte Putra-Bestand liegt zeitlich vor PhreshPhish 2025.', '']
for tag,title in [('direct','Quell-RUT gegenüber R0'),('target','Zusätzliche Zielanpassung E1 gegenüber Quell-RUT E0')]:
    r=read(f'empirical_putra_{tag}_primary.csv').iloc[0]
    S += ['### '+title,'',f'AP-AULC: {f(100*r.baseline_mean)} % → {f(100*r.adapted_mean)} %. Δ = **{f(100*r.mean_delta)} pp**, 95-%-KI [{f(100*r.ci95_lo)}; {f(100*r.ci95_hi)}] pp; p = {p(r.p_raw)}.', '']
    b=read(f'empirical_putra_{tag}_budgets.csv')
    S += [table(['Labels','AP Basis (%)','AP angepasst (%)','Δ AP (pp)','95-%-KI (pp)','positive/negative Paare','p roh','p Holm'],[[r.budget,f(100*r.baseline_mean),f(100*r.adapted_mean),f(100*r.mean_delta),f'[{f(100*r.ci95_lo)}; {f(100*r.ci95_hi)}]',f'{r.positive_pairs}/{r.negative_pairs}',p(r.p_raw),p(r.p_holm)] for r in b.itertuples()]),'']
S += ['Die beiden vierteiligen Budgetfamilien werden separat korrigiert. Die AP-AULC-Hauptvergleiche sind keine fünften Budgettests. Der positive Transferbefund und die negative weitere Zielanpassung betreffen unterschiedliche Interventionen.', '',
'120 neue Logistic-Regression-Fits aus wiedergewonnenen Embeddings und identischen Auswahlhashes ergeben ΔAP-AULC +2,535073 pp bzw. −1,138777 pp; beide p=0,001953125. Die historischen Werte bleiben als ursprüngliche Versuchsergebnisse erhalten. Von 480 Metrikvergleichen weichen 113 bei einer Toleranz von 10⁻⁸ ab. Größte Abweichungen: AP 0,005491 pp, AUC 0,001691 pp, FPR@TPR90 0,059312 pp, P≥90 0,086337 pp. Ursachen können Solver-/Umgebungsunterschiede sein; eine konkrete Ursache wurde nicht isoliert.', '',
'## FF4: Komponenten, finale Abstände, Laufzeit und Eskalation', '',
'Sequenzielle DEV-Komponentenablation, N10, gleiche Labels und gepaarte Seeds:', '']
a=read('empirical_ff4_components.csv')
S += [table(['Variante','TPR (%)','FPR (%)','FPR@TPR90 (%)'],[[r.variant,f(100*r.tpr,3),f(100*r.fpr,3),f(100*r.FPR_at_TPR90,3)] for r in a.itertuples()]),'']
a=read('empirical_ff4_contrasts.csv')
S += [table(['Kontrast','Δ TPR (pp)','95-%-KI (pp)','p roh','p Holm'],[[r.contrast,f(100*r.mean_delta,3),f'[{f(100*r.ci95_lo,3)}; {f(100*r.ci95_hi,3)}]',p(r.p_raw),p(r.p_holm)] for r in a.itertuples()]),'',
'Beim Gating sind neun Differenzen positiv und eine null; deshalb p roh=0,00390625. Holm ist für alle drei Kontraste 0,005859375. Diese nachträgliche Komponentenanalyse ist von der eingefrorenen FINAL-Systemauswertung zu unterscheiden.', '',
'Finale Systemabstände aus TP/FP-Zählungen; deskriptive Einzelmodellvergleiche:', '']
a=read('closure_ff4_system_gaps.csv')
S += [table(['Bedingung','Kontrast','Δ TPR (pp)','Δ FPR (pp)'],[[r.scenario,r.contrast,f(r.tpr_delta_pp),f(r.fpr_delta_pp)] for r in a.itertuples()]),'']
a=read('empirical_ff4_system.csv');a=a[a.scenario.eq('OFFICIAL_TEST')]
S += [table(['Modus','TP','FP','TPR (%)','FPR (%)','AP (%)','FPR@TPR90 (%)'],[[r.mode,r.tp,r.fp,f(100*r.tpr),f(100*r.fpr),f(100*r.AP),f(100*r.FPR_at_TPR90)] for r in a.itertuples()]),'']
b=read('raw_bootstrap_summary.csv')
S += ['Rohscore-Reproduktion der Rankingmetriken mit je 1.000 stratifizierten Bootstrap-Stichproben:', '',table(['System','AP (%)','95-%-Percentile-KI AP (%)','P≥90 (%)','95-%-Percentile-KI P≥90 (%)'],[[r.system,f(100*r.AP,6),f'[{f(100*r.AP_ci95_lo,6)}; {f(100*r.AP_ci95_hi,6)}]',f(100*r.P_at_R90,6),f'[{f(100*r.P_at_R90_ci95_lo,6)}; {f(100*r.P_at_R90_ci95_hi,6)}]'] for r in b.itertuples()]),'']
r=read('empirical_runtime_derived.csv').iloc[0]
S += [f'DEV-In-Memory-Benchmark: Median {f(r.full_latency_ms,6)} → {f(r.cascade_latency_ms,6)} ms; Quotient Full/Kaskade **{f(r.full_latency_ms/r.cascade_latency_ms,6)}**, entsprechend **{f(r.latency_reduction_pct,6)} %** geringerer Median-Latenz. Durchsatz {f(r.full_pages_s,6)} → {f(r.cascade_pages_s,6)} Seiten/s; Faktor **{f(r.speedup,6)}** bzw. {f(100*(r.speedup-1),4)} % Zunahme.', '',
'95. Latenzperzentil: 47,402349 → 32,136478 ms; Durchsatz-SD über drei Läufe: 1,32461 → 2,95146 Seiten/s. Die absoluten Zeitquantile und Streuungen sind gegen Benchmark-Code und gespeicherte Aggregate geprüft. Rohzeitreihen fehlen; diese Werte wurden nicht unabhängig neu gemessen. Keine Netzwerk-, Rendering-, Datenträger- oder vorgelagerten HTML-Parsingkosten enthalten.', '']
b=pd.read_csv(ROOT/'artifacts/CH6_ANALYSIS_FULL/thesis_tables/TABLE_OPERATIONAL_BENCHMARK.csv');full=b[b.system.eq('TRI_RUT_200K_FULL')].iloc[0]
S += ['Alle vier gespeicherten Benchmarkpfade; Quotienten aus ungerundeten Aggregaten:', '',table(['System','Median (ms)','p95 (ms)','Seiten/s','Median-Faktor Full/Variante','p95-Faktor Full/Variante','Durchsatz-Faktor Variante/Full'],[[r.system,f(r.latency_b1_median_ms,4),f(r.latency_b1_p95_ms,4),f(r.throughput_pages_s_mean,4),f(full.latency_b1_median_ms/r.latency_b1_median_ms,6),f(full.latency_b1_p95_ms/r.latency_b1_p95_ms,6),f(r.throughput_pages_s_mean/full.throughput_pages_s_mean,6)] for r in b.itertuples()]),'']
r=read('empirical_routing.csv').iloc[0]
S += [f'Eingefrorene FINAL-Kaskade: {int(r.escalated_n)} / {int(r.final_n)} = **{f(100*r.escalation_rate,6)} % Eskalation**, {f(100*r.early_exit_rate,6)} % frühe Entscheidungen. Historische Float32-Grenzvergleiche wurden explizit reproduziert; eine veränderte Float64-Grenzbehandlung würde {f(100*r.float64_alternative_escalation_rate,6)} % erzeugen und wäre ein anderes Routing.', '',
'Separater N5-Frontendvergleich auf Official, alle vier DEV-Toleranzen:', '']
a=read('empirical_ff4_frontend_official.csv')
S += [table(['Toleranz (pp)','Frontend','Eskalation (%)','frühe Entscheidung (%)','Δ TPR zu Full (pp)','Δ FPR zu Full (pp)'],[[f(r.allowed_tpr_loss_pp,2),r.stream.replace('DAPT','TAPT'),f(100*r.escalation_rate),f(100*r.full_path_avoided_rate),f(r.delta_tpr_vs_full_pp),f(r.delta_fpr_vs_full_pp)] for r in a.itertuples()]),'',
'Bei 1 pp sinkt Official-Eskalation von 35,746043 % auf 23,513864 % (−12,232179 pp). Der ungewichtete Mittelwert über fünf überlappende Szenarien beträgt dagegen 41,124611 % → 28,727791 % (−12,396819 pp). Er ist keine Populationsrate. Beide Werte gehören zu einem anderen Routingprotokoll als die eingefrorenen 14,933357 %.', '',
'## Base-Rate-Umrechnung', '',
'P≥90 bezeichnet die höchste empirische Precision unter Schwellen mit Recall ≥0,9. FPR@TPR90 verwendet dagegen den ersten ROC-Punkt mit TPR ≥0,9. Beide Größen dürfen nicht ohne Schwellenabgleich kombiniert werden.', '',
'Mit π₀=76.800/168.060=0,4569796501249554 und der ungerundeten P≥90-Precision gilt λ=((1−P₀)/P₀)·π₀/(1−π₀)=FPR*/TPR*. Daraus folgt Precision(π)=π/[π+(1−π)λ]. Die unabhängige äquivalente Bayes-Rechnung liefert dieselben Werte.', '']
a=read('empirical_base_rate.csv')
S += [table(['System','Prävalenz (%)','Precision (%)','Schlussfassung gerundet (%)'],[[r.system,f(100*r.prevalence,6),f(100*r.precision,6),f(100*r.precision,3 if r.prevalence>.1 else 2)] for r in a.itertuples()]),'',
'Korrektur der letzten noch abweichenden Text-/Tabellenwerte: bei 5 % 95,61→95,62 % und 96,53→96,54 %; bei 1 % 80,71→80,74 % und 84,24→84,27 %. Die Umrechnung gilt bei konstantem klassenspezifischem Scoreverhalten und ist kein externer empirischer Benchmark.', '',
'## Reproduktionsweg', '',
'`Reproduktion.zip` entpacken; dort `README.md` folgen. Hauptskripte: `empirical_audit.py`, `raw_score_audit.py`, `closure_audit.py`. Dieses Protokoll wird mit `core_report.py` aus den geprüften CSVs erzeugt. Sign-Flip-p-Werte beruhen auf vollständiger Vorzeichenenumeration, setzen für die inferenzielle Interpretation jedoch Symmetrie/Austauschbarkeit voraus. t-Intervalle beschreiben die Variation der gegebenen Replikationen; sie ersetzen keine unabhängigen Daten- oder Vortrainingswiederholungen.', '']
text='\n'.join(S)
(O/'Kernpruefung.md').write_text(text,encoding='utf-8')
(R/'KERNPRUEFUNG.md').write_text(text.replace('../review/QUELLENPRUEFUNG.md','QUELLENPRUEFUNG.md'),encoding='utf-8')
print('Kernpruefung.md:',len(text),'Zeichen')
