"""Original three-panel scientific figure from frozen data and manuscript method."""
from pathlib import Path
import csv,json,hashlib,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'audits/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

def main():
    for name,info in json.loads((ROOT/'source_data/manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/'source_data'/name).read_bytes()).hexdigest()==info['sha256']
    rows=list(csv.DictReader((ROOT/'source_data/synthetic_summary.csv').open()))
    values={}
    for cond in ['ideal','lowpass_200Hz']:
        values[cond]=[]
        for method in ['spatial','corrected']:
            rr=[r for r in rows if r['part']=='B' and r['observation']==cond and r['method']==method]
            assert len(rr)==1 and int(rr[0]['target_n'])==96
            values[cond].append(float(rr[0]['target_macro_accuracy'])*100)
    spec=list(csv.DictReader((ROOT/'source_data/per_specimen_effects.csv').open()))
    ids=[int(r['specimen_id']) for r in spec];assert ids==list(range(12))
    dif=[float(r['difference_pp']) for r in spec]
    ex=json.loads((ROOT/'source_data/external_summary.json').read_text())
    mean=ex['all12_difference_pp'];lo,hi=ex['original_conditional_interval_pp']
    assert abs(sum(dif)/12-mean)<1e-12
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':6.5,
        'ytick.labelsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'legend.frameon':False})
    blue='#3175A5';orange='#D48B46';dark='#30343B'
    fig=plt.figure(figsize=(7,1.35))
    # a: conceptual schematic of implemented processing; no generated signal traces.
    a=fig.add_axes([.005,.05,.315,.88]);a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off')
    fig.text(.007,.965,'(a) Source-only calibration',fontsize=8,weight='bold',va='top')
    def box(x,y,w,h,label,face,edge):
        a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.025',fc=face,ec=edge,lw=.65))
        a.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=6.5,color=dark,linespacing=1.15)
    for x,label,color in [(.015,'SOURCE',blue),(.535,'TARGET',orange)]:
        a.text(x+.21,.84,label,ha='center',fontsize=6.5,color=color,weight='bold')
    source=['Spatial log spectra\n+ speeds / labels','Class balance\n+ double centering','Cosine ridge fit\n(frozen coefficients)']
    target=['Spatial log spectra\n+ known speed','Subtract fitted term\n+ renormalize','Source-fitted\nscaler + SVM']
    for labels,x,face,edge in [(source,.015,'#EAF2F8',blue),(target,.535,'#FCF2E7',orange)]:
        for y,label in zip([.61,.36,.11],labels):box(x,y,.425,.19,label,face,edge)
        for y in [.60,.35]:a.annotate('',xy=(x+.2125,y-.045),xytext=(x+.2125,y),arrowprops={'arrowstyle':'->','lw':.7,'color':dark})
    a.plot([.445,.49,.49],[.20,.20,.46],color=blue,lw=.85)
    a.annotate('',xy=(.53,.46),xytext=(.49,.46),arrowprops={'arrowstyle':'->','lw':.85,'color':blue,'shrinkA':0,'shrinkB':0})
    a.text(.5,.00,'Target labels / distribution excluded from fitting',ha='center',fontsize=6.2,color=dark)
    # b: fixed synthetic comparison. No between-run interval was estimated.
    b=fig.add_axes([.39,.24,.235,.59])
    fig.text(.365,.965,'(b) Synthetic observations',fontsize=8,weight='bold',va='top')
    for k,(method,color,hatch) in enumerate([('Spatial',orange,''),('Calibrated',blue,'///')]):
        heights=[values[c][k] for c in ['ideal','lowpass_200Hz']]
        xpos=[i+(k-.5)*.32 for i in range(2)]
        b.bar(xpos,heights,width=.30,color=color,edgecolor='white',lw=.5,hatch=hatch,label=method,zorder=3)
        for x,y in zip(xpos,heights):b.text(x,y+2,f'{y:.2f}',ha='center',va='bottom',fontsize=6)
    b.set_ylim(0,115);b.set_yticks([0,50,100]);b.set_ylabel('Accuracy (%)',labelpad=1)
    b.set_xticks([0,1],['Ideal','Low-pass']);b.tick_params(length=2,pad=2)
    b.grid(axis='y',color='.90',lw=.5,zorder=0)
    b.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=6.2,handlelength=1,handletextpad=.3,columnspacing=.65,borderaxespad=0)
    # c: all fixed specimens and macro interval; no ordering by outcome.
    c=fig.add_axes([.70,.24,.29,.59])
    fig.text(.68,.965,'(c) External specimen effects',fontsize=8,weight='bold',va='top')
    c.bar(ids,dif,width=.72,color=blue,edgecolor='white',lw=.4,zorder=3)
    c.scatter([i for i,d in zip(ids,dif) if d==0],[0]*sum(d==0 for d in dif),marker='_',color=blue,s=20,zorder=4)
    c.axhline(0,color=dark,lw=.7,zorder=2)
    c.axvline(12.1,color='.8',lw=.6)
    c.errorbar([13.2],[mean],yerr=[[mean-lo],[hi-mean]],fmt='D',ms=3.8,color=dark,elinewidth=1,capsize=2.5,zorder=5)
    c.set_xlim(-.7,14.2);c.set_ylim(-32,24);c.set_yticks([-30,-15,0,15]);c.set_ylabel('Calibrated − spatial (pp)',labelpad=1)
    c.set_xticks(ids+[13.2],[str(i) for i in ids]+['All']);c.tick_params(length=2,pad=2)
    c.grid(axis='y',color='.92',lw=.5,zorder=0)
    c.set_xlabel('Specimen ID; all retained',labelpad=2)
    fig.savefig(ROOT/'figures/overview.pdf')
    fig.savefig(ROOT/'figures/overview.svg')
    fig.savefig(ROOT/'figures/overview.png',dpi=600)
    plt.close(fig)
    (ROOT/'source_data/figure_values.json').write_text(json.dumps({'synthetic_accuracy_pct':values,'specimen_ids':ids,'specimen_difference_pp':dif,'macro_difference_pp':mean,'macro_conditional_interval_pp':[lo,hi]},indent=2)+'\n')

if __name__=='__main__':main()
