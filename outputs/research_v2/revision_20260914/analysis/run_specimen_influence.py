#!/usr/bin/env python3
"""Descriptive post-hoc analysis; never refits, excludes, or changes benefit rule."""
from pathlib import Path
import csv, json, hashlib, platform, sys, time, os
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'matplotlib_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    start=time.perf_counter(); utc=datetime.now(timezone.utc).isoformat()
    plan=json.loads((OUT/'analysis_plan.json').read_text())
    assert plan['created_utc'] < utc
    inputs=[ROOT/p for p in plan['inputs']]
    report=json.loads(inputs[2].read_text()); prior=report['external_primary']
    checks={}
    for p in inputs[:2]:
        key=str(p.relative_to(ROOT)); checks[key+'_hash']=sha(p)==report['input_sha256'][key]
    assert all(checks.values())
    def load(p):
        rows=list(csv.DictReader(p.open()))
        for r in rows:
            for k in ['index','texture','speed','prediction','correct']: r[k]=int(r[k])
            assert r['correct']==int(r['texture']==r['prediction'])
            assert r['speed']==40 and r['domain']=='speed'
        assert len({r['index'] for r in rows})==len(rows)
        assert len({r['acquisition_id'] for r in rows})==len(rows)
        return sorted(rows,key=lambda r:r['index'])
    a,b=map(load,inputs[:2]); keys=['index','acquisition_id','texture','speed','domain']
    assert len(a)==len(b)==1196
    assert all(all(x[k]==y[k] for k in keys) for x,y in zip(a,b))
    ids=sorted({r['texture'] for r in a}); assert len(ids)==12
    checks.update(unique_keys=True,exact_pairing=True,correct_fields_verified=True,target_rows_1196=True,all_12_specimens=True)
    rows=[]
    for t in ids:
        aa=[r for r in a if r['texture']==t]; bb=[r for r in b if r['texture']==t]
        n=len(aa); ca=sum(r['correct'] for r in aa); cb=sum(r['correct'] for r in bb)
        ra=100*ca/n; rb=100*cb/n; delta=100*(ca-cb)/n
        rows.append(dict(specimen_id=t,n=n,corrected_correct=ca,spatial_correct=cb,corrected_recall_pct=ra,spatial_recall_pct=rb,difference_pp=delta,contribution_to_12_specimen_macro_pp=delta/12))
    delta=float(np.mean([r['difference_pp'] for r in rows])); assert abs(delta-prior['difference_pp'])<1e-12
    for r in rows:
        loo=float(np.mean([x['difference_pp'] for x in rows if x is not r]))
        r['leave_one_specimen_out_difference_pp']=loo
        r['change_from_all12_pp']=loo-delta
        r['removal_changes_sign']=bool(np.sign(loo)!=np.sign(delta))
    signs=dict(improved=sum(r['difference_pp']>0 for r in rows),worsened=sum(r['difference_pp']<0 for r in rows),tied=sum(r['difference_pp']==0 for r in rows))
    assert signs==dict(improved=6,worsened=3,tied=3)
    with (OUT/'per_specimen_effects.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'created_utc':utc,'status':'exploratory post-hoc descriptive influence analysis','pairing_checks':checks,'n_target_acquisitions':len(a),'n_specimens':12,'corrected_macro_recall_pct':float(np.mean([r['corrected_recall_pct'] for r in rows])),'spatial_macro_recall_pct':float(np.mean([r['spatial_recall_pct'] for r in rows])),'all12_difference_pp':delta,'original_conditional_interval_pp':[prior['conditional_low_pp'],prior['conditional_high_pp']],'interval_scope':report['interval_scope'],'interval_provenance':'Copied unchanged from frozen analysis/report.json; not recalculated or reinterpreted.','specimen_signs':signs,'leave_one_specimen_out_range_pp':[min(r['leave_one_specimen_out_difference_pp'] for r in rows),max(r['leave_one_specimen_out_difference_pp'] for r in rows)],'removals_changing_sign':[r['specimen_id'] for r in rows if r['removal_changes_sign']],'original_primary_support':report['external_primary_support'],'original_rule':'At least 3 pp improvement with positive conditional lower bound; unchanged.','limitations':['No independent new-instance or new-session test.','Acquisitions within a physical specimen are not independent specimens.','LOSO is influence only; all 12 retained in primary comparison.','Observed sign changes are not evidence to exclude specimens.','No new p-values, refits, thresholds or application gate.']}
    dump(OUT/'summary.json',summary)
    plt.rcParams.update({'font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(3.45,2.7),sharey=True,gridspec_kw={'width_ratios':[1.1,1]})
    y=np.arange(12); dif=np.array([r['difference_pp'] for r in rows]); loo=np.array([r['leave_one_specimen_out_difference_pp'] for r in rows])
    axes[0].hlines(y,0,dif,color='0.6',linewidth=1)
    axes[0].scatter(dif,y,s=16,color='#245B88',zorder=3)
    axes[1].scatter(loo,y,s=16,color='#245B88',zorder=3)
    axes[1].axvline(delta,color='#9B4B12',ls='--',lw=.9,label='All 12')
    for ax in axes:
        ax.axvline(0,color='0.25',lw=.65);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(11.6,-.6)
    axes[0].set_yticks(y,[str(t) for t in ids]);axes[0].set_ylabel('Specimen ID')
    axes[0].set_title('(a) Paired recall difference');axes[1].set_title('(b) Omit indicated specimen')
    axes[0].set_xlabel('Calibrated − spatial (pp)');axes[1].set_xlabel('Panel mean difference (pp)')
    axes[1].legend(frameon=False,fontsize=6,loc='lower right')
    fig.tight_layout(pad=.5,w_pad=.8)
    fig.savefig(OUT/'specimen_influence.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(OUT/'specimen_influence.svg',metadata={'Date':None})
    fig.savefig(OUT/'specimen_influence.png',dpi=220)
    plt.close(fig)
    tex=['% Generated from saved paired predictions. Exploratory influence only.',r'\begin{tabular}{rrrrrr}',r'\hline',r'ID & $n$ & Corrected & Spatial & $\Delta$ (pp) & Omit ID (pp) \\',r'\hline']
    for r in rows: tex.append(f"{r['specimen_id']} & {r['n']} & {r['corrected_correct']} & {r['spatial_correct']} & {r['difference_pp']:+.2f} & {r['leave_one_specimen_out_difference_pp']:+.2f} \\\\")
    tex += [r'\hline',r'\end{tabular}']
    (OUT/'specimen_influence_table.tex').write_text('\n'.join(tex)+'\n')
    macros={'SpecimenImproved':signs['improved'],'SpecimenWorsened':signs['worsened'],'SpecimenTied':signs['tied'],'InfluenceLow':f"{summary['leave_one_specimen_out_range_pp'][0]:.2f}",'InfluenceHigh':f"{summary['leave_one_specimen_out_range_pp'][1]:.2f}",'InfluenceAllPanel':f'{delta:.2f}'}
    (OUT/'specimen_influence_numbers.tex').write_text('% Generated; exploratory post-hoc influence analysis.\n'+''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in macros.items()))
    (OUT/'README.md').write_text('# External specimen influence: exploratory\n\nRun from the project root:\n\n```sh\nwork/research/venv/bin/python outputs/research_v2/revision_20260914/analysis/run_specimen_influence.py\n```\n\nAll 1,196 paired target acquisitions from all 12 fixed specimens are retained. Corrected and spatial columns in the table are **correct counts**, not percentages; full recalls and contributions are in the CSV. The paired difference is a difference in specimen recalls, in percentage points. Contributions divide each difference by 12. The figure shows every specimen in ID order, with no selection. Its right panel omits each indicated specimen solely to quantify influence; the dashed line is the unchanged all-12 mean. No uncertainty intervals are added to per-specimen effects because acquisitions are repeated measurements of these specimens.\n\nThe original conditional interval is copied from the previously frozen analysis, not newly estimated. A majority of positive specimen differences does not imply a positive mean because differences have unequal magnitudes. No specimen is removed from the main conclusion, and the original benefit criterion remains unchanged. This analysis cannot establish harm, equivalence, future-specimen performance or a physical failure mechanism.\n')
    env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'matplotlib':matplotlib.__version__}
    dump(OUT/'environment.json',env)
    log={'utc':utc,'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,'status':'completed','checks':checks,'original_inputs_unchanged':all(sha(p)==report['input_sha256'][str(p.relative_to(ROOT))] for p in inputs[:2])}
    dump(OUT/'run_log.json',log)
    manifest={'utc':log['finished_utc'],'command':'work/research/venv/bin/python outputs/research_v2/revision_20260914/analysis/run_specimen_influence.py','inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'script_sha256':sha(Path(__file__)),'outputs':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ['manifest.json','freeze.json']}}
    dump(OUT/'manifest.json',manifest)
    dump(OUT/'freeze.json',{'utc':log['finished_utc'],'status':'complete; pending independent parent verification','files':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='freeze.json'}})
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
