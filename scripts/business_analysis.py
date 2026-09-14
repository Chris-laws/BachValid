"""AI-assisted operational scenario analysis; no new training or threshold selection.

Run from any directory. --require-raw requires recovered FINAL labels; the public
repository otherwise uses the explicitly labelled, independently audited counts.
All prevalence/volume scenarios are assumptions, not observations at the employer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / 'artifacts/CH6_ANALYSIS_FULL'
OUT = ROOT / 'review'
OUT.mkdir(exist_ok=True)
args = argparse.ArgumentParser(description=__doc__)
args.add_argument('--require-raw', action='store_true')
args = args.parse_args()
checks, hashes = [], {}


def track(p):
    hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def check(name, actual, expected, tol=1e-12):
    ok = bool(np.isclose(actual, expected, atol=tol, rtol=0))
    checks.append(dict(check=name, actual=float(actual), expected=float(expected), passed=ok))
    assert ok, (name, actual, expected)


table = pd.read_csv(track(CH / 'thesis_tables/TABLE_FF4_DOM_FUSION_CASCADE_FINAL.csv'))
q = table[table.system.eq('TRI_RUT_200000') & table.scenario.eq('OFFICIAL_TEST')
          & np.isclose(table.target_fpr, .005)].set_index('mode')
freeze = json.loads(track(CH / 'audit/ARCHITECTURE_FREEZE.json').read_text())
check('Full system training budget', freeze['selected_budget'], 200000)
bench = pd.read_csv(track(CH / 'thesis_tables/TABLE_OPERATIONAL_BENCHMARK.csv')).set_index('system')
mapping = [('URL', 'url', 'URL_ONLY_FROM_TRI_RUT'),
           ('FULL', 'gated', 'TRI_RUT_200K_FULL'),
           ('CASCADE', 'cascade', 'TRI_RUT_200K_CASCADE')]
raw = ROOT / 'artifacts/raw_recovery/role_meta/FINAL.parquet'
cache = OUT / 'business_routing_counts.csv'
if raw.exists():
    y = pd.read_parquet(track(raw)).y.to_numpy(dtype=int)
    check('FINAL labels binary', np.isin(y, [0, 1]).all(), True)
    sdir = CH / 'scores/ff4/TRI_RUT_200000'
    u = np.load(track(sdir / 'FINAL_url.npy')).astype(np.float32)
    g = np.load(track(sdir / 'FINAL_gated.npy')).astype(np.float64)
    check('URL score count', len(u), len(y))
    check('Gated score count', len(g), len(y))
    check('URL scores finite', np.isfinite(u).all(), True)
    check('Full scores finite', np.isfinite(g).all(), True)
    # Match the notebook's NumPy 1.x float32 scalar-comparison/tie semantics.
    low = u <= np.float32(freeze['cascade']['url_low'])
    high = u >= np.float32(freeze['cascade']['url_high'])
    esc = ~(low | high)
    c = g.copy()
    c[low], c[high] = -1e9, 1e9
    for name, mode, _ in mapping:
        s = {'url': u.astype(float), 'gated': g, 'cascade': c}[mode]
        pred = s >= q.loc[mode, 'threshold']
        counts = dict(tp=np.sum(pred & (y == 1)), fp=np.sum(pred & (y == 0)),
                      tn=np.sum(~pred & (y == 0)), fn=np.sum(~pred & (y == 1)))
        for key, value in counts.items():
            check(name + ' raw ' + key, value, q.loc[mode, key])
    routing = pd.DataFrame([dict(label=k, n=int(np.sum(y == k)),
                                  escalated=int(np.sum(esc & (y == k)))) for k in [0, 1]])
    routing.to_csv(cache, index=False)
    input_mode = 'raw FINAL labels and scores; all three confusion matrices checked'
else:
    if args.require_raw:
        raise FileNotFoundError('--require-raw: missing ' + str(raw))
    routing = pd.read_csv(track(cache))
    input_mode = 'cached audited class-conditional routing counts; raw labels unavailable'
routing = routing.set_index('label')
check('Benign population', routing.loc[0, 'n'], q.loc['gated', 'fp'] + q.loc['gated', 'tn'])
check('Phishing population', routing.loc[1, 'n'], q.loc['gated', 'tp'] + q.loc['gated', 'fn'])
e0, e1 = (float(routing.loc[k, 'escalated'] / routing.loc[k, 'n']) for k in [0, 1])
prior0 = float(routing.loc[1, 'n'] / routing.n.sum())
check('Routing class-weighted reconstruction', (1-prior0)*e0+prior0*e1,
      routing.escalated.sum()/routing.n.sum())
ops, resources, scenarios = [], [], []
for name, mode, system in mapping:
    r, b = q.loc[mode], bench.loc[system]
    tpr = float(r.tp / (r.tp+r.fn))
    fpr = float(r.fp / (r.fp+r.tn))
    check(name + ' saved TPR', tpr, r.tpr)
    check(name + ' saved FPR', fpr, r.fpr)
    ops.append(dict(system=name, training_labels=200000, threshold=float(r.threshold),
                    tp=int(r.tp), fp=int(r.fp), tn=int(r.tn), fn=int(r.fn), tpr=tpr, fpr=fpr))
    resources.append(dict(system=name, median_ms=float(b.latency_b1_median_ms),
                          p95_ms=float(b.latency_b1_p95_ms), pages_s=float(b.throughput_pages_s_mean),
                          reference_seconds_10000=10000/float(b.throughput_pages_s_mean),
                          hardware=b.hardware, scope=b.measurement_scope))
    for prior in [.001, .01, .05, prior0]:
        n = 10000
        tp, fp, fn, tn = n*prior*tpr, n*(1-prior)*fpr, n*prior*(1-tpr), n*(1-prior)*(1-fpr)
        er = (1-prior)*e0+prior*e1 if name == 'CASCADE' else float(name == 'FULL')
        row = dict(system=name, volume=n, prevalence=prior, tp=tp, fp=fp, fn=fn, tn=tn,
                   model_positive=tp+fp, precision=tp/(tp+fp),
                   full_path_rate=er, full_path_cases=n*er)
        scenarios.append(row)
        check(name + f' prior={prior:g} population conservation', tp+fp+fn+tn, n, 1e-9)
        check(name + f' prior={prior:g} positive conservation', tp+fn, n*prior, 1e-9)
        # Direct class-reweighting of empirical counts provides a second calculation.
        w1, w0 = n*prior/(r.tp+r.fn), n*(1-prior)/(r.fp+r.tn)
        check(name + f' prior={prior:g} weighted precision', row['precision'], r.tp*w1/(r.tp*w1+r.fp*w0))
for file, rows in [('business_operating_points.csv', ops), ('business_resources.csv', resources),
                   ('business_scenarios.csv', scenarios), ('business_checks.csv', checks)]:
    pd.DataFrame(rows).to_csv(OUT/file, index=False)
data = dict(input_mode=input_mode, reference_prevalence=prior0,
            routing=dict(benign_rate=e0, phishing_rate=e1,
                         benign_count=int(routing.loc[0, 'n']), phishing_count=int(routing.loc[1, 'n']),
                         benign_escalated=int(routing.loc[0, 'escalated']),
                         phishing_escalated=int(routing.loc[1, 'escalated'])),
            operating_points=ops, resources=resources,
            assumptions='Prior shift only; fixed class-conditional error and routing distributions. Volumes and priors are illustrative. Benchmark time is not end-to-end or reweighted timing.')
(OUT/'business_inputs.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT/'business_input_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
summary = dict(checks=len(checks), passed=sum(x['passed'] for x in checks), input_mode=input_mode)
(OUT/'business_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

def de(v, digits=2):
    return f'{v:.{digits}f}'.replace('.', ',')

lines = ['# Betriebskonzept: Rechen- und Herkunftsprüfung', '',
         'Die Szenarien sind eine nachträgliche Interpretation der eingefrorenen Ergebnisse. Sie sind keine Messungen bei der Volksbank und kein neuer Systemtest.', '',
         f"Prüfmodus: {input_mode}. {len(checks)} Prüfungen bestanden.", '',
         '## Operative Fallzahlen', '',
         'Je 10.000 klassifizierbare Webseiten; gleiche klassenspezifische Fehler wie auf OFFICIAL_TEST. Alle drei Betriebsarten stammen aus TRI_RUT_200000, nicht aus FF3-RUT50k.', '',
         '| Phishinganteil | Modell | TP | FP | FN | Modellpositiv | Precision | zusätzliche Vollpfade |',
         '|---|---|---:|---:|---:|---:|---:|---:|']
for r in sorted(scenarios, key=lambda x:(x['prevalence'],x['system'])):
    lines.append('| '+ ' | '.join([de(100*r['prevalence'])+' %',r['system']]
                 +[de(r[k]) for k in ['tp','fp','fn','model_positive']]
                 +[de(100*r['precision'])+' %',de(r['full_path_cases'])])+' |')
lines += ['', '## Entscheidungsrelevante Trennungen', '',
          '- Die Precision hier gilt an der auf CAL bestimmten Schwelle. Die vorhandene P≥90-Prävalenzrechnung benutzt eine andere empirische Schwelle. Ihre 80,74/84,27 % bei 1 % dürfen nicht mit den hiesigen Betriebsraten kombiniert werden.',
          f'- Kaskadenrouting aus Rohscores: {routing.loc[0,"escalated"]:.0f}/{routing.loc[0,"n"]:.0f} benign ({de(100*e0,3)} %) und {routing.loc[1,"escalated"]:.0f}/{routing.loc[1,"n"]:.0f} Phishing ({de(100*e1,3)} %). Bei verändertem Prior gilt e(π)=(1−π)e0+πe1. 14,933 % gilt nur für die ursprüngliche Mischung.',
          '- Zusätzlicher Vollpfad, Modellalarm und menschlicher Prüffall sind verschiedene Größen. Bei Prüfung aller Modellpositiven ist TP+FP nur der entsprechende Teil der Prüfmenge; Negativstichproben, nicht auswertbare Fälle und weiterer Kontext kommen hinzu.',
          '- In-Memory-Referenzzeit = 10.000 / gemessener mittlerer Durchsatz. Keine End-to-End-Prognose und keine Laufzeitprognose für eine geänderte Prävalenz. Einzelzeitreihen und klassenspezifische Zeitmessungen fehlen.',
          '- 50k/75 % ist der Nachweis für das interne Linear-Probe-Training. Die hier betrachteten Systemmodelle nutzten 200k Trainingslabels. Kalibrations-, Entwicklungs- und Evaluationslabels sowie Erhebung und TAPT sind getrennte Ressourcen.',
          '- Der vorgeschlagene Einsatzprozess und seine Freigabekriterien sind eigene Konzeption. Die Umsetzung von Erfassung, Tickets, Menschprüfung und Monitoring wurde nicht experimentell validiert.', '',
          '## Quellenbezug', '',
          'NIST SP 800-61r3 (Nelson, Rekhi, Souppaya, Scarfone; April 2025), gedruckte S. 25: technische Filterung für menschliche Analyse; S. 26–27: kontextbezogene Priorisierung und Validierung von Meldungen. Original geprüft: https://doi.org/10.6028/NIST.SP.800-61r3. Daraus folgt keine NIST-Empfehlung für dieses konkrete Modell.', '',
          '## Reproduktion', '',
          '`python scripts/business_analysis.py --require-raw` prüft die Rohscores/Labels erneut. Ohne diesen Parameter und ohne Rohlabels verwendet das öffentliche Paket `review/business_routing_counts.csv`; dieser Modus ist ausdrücklich im Ergebnis ausgewiesen. Alle Eingabedateien werden gehasht. Die Berechnung verändert weder Modelle noch historische Ergebnisartefakte.', '']
(OUT/'BETRIEBSKONZEPT_PRUEFUNG.md').write_text('\n'.join(lines), encoding='utf-8')
out = ROOT/'Abgabe'
out.mkdir(exist_ok=True)
template = (ROOT/'scripts/business_calculator.html').read_text(encoding='utf-8')
(out/'Betriebsrechner.html').write_text(template.replace('__AUDITED_DATA__', json.dumps(data, ensure_ascii=False)), encoding='utf-8')
print(json.dumps(summary), flush=True)
