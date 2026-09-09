"""Analyze saved F/T trials and write CSV/JSON plus dependency-light SVG figures."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from ablation_metrics import metrics, process_stream


def scalar(value):
    return value if value is None or isinstance(value,(str,bool,int,float)) else json.dumps(value)


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False))


def line_plot(path,t,series,title):
    colors=['#1565c0','#c62828','#2e7d32','#7b1fa2','#ef6c00','#00838f']
    width,height=850,240*len(series)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d"><rect width="100%%" height="100%%" fill="white"/>'%(width,height),
           '<text x="15" y="22" font-family="sans-serif">'+html.escape(title)+'</text>']
    for panel,(label,values) in enumerate(series):
        values=np.asarray(values);values=values[:,None] if values.ndim==1 else values
        lo=float(values.min());hi=float(values.max());span=max(hi-lo,1e-6)
        top=50+panel*240
        parts.append('<text x="15" y="%d" font-family="sans-serif">%s [%.3g, %.3g]</text>'%(top,html.escape(label),lo,hi))
        parts.append('<path d="M60 %d v160 h750" stroke="black" fill="none"/>'%(top+15))
        for j in range(values.shape[1]):
            xy=np.column_stack([60+750*(t-t[0])/max(t[-1]-t[0],1e-9),top+175-160*(values[:,j]-lo)/span])
            parts.append('<polyline fill="none" stroke="%s" stroke-width="1.5" points="%s"/>'%(colors[j%6],' '.join('%.2f,%.2f'%tuple(x) for x in xy)))
        parts.append('<text x="60" y="%d" font-family="sans-serif">%.1f s → %.1f s; channels: blue/red/green/purple/orange/teal</text>'%(top+195,t[0],t[-1]))
    path.write_text(''.join(parts)+'</svg>')


def summarize(rows,out):
    # Aggregate whole trials, not overlapping fit windows or sensor samples.
    numeric=['apparent_inertia','axial_effort_rms','breakaway_peak','onset_delay_s',
             'rotation_leak_deg_per_m','post_release_translation_m','post_release_rotation_deg',
             'settling_time_s','core_saturation_fraction','translation_positive_work_J']
    summary=[];rng=np.random.default_rng(42)
    synthetic_label='SYNTHETIC DEMO — NOT ROBOT RESULTS — ' if any(r.get('synthetic') for r in rows) else ''
    keys=sorted(set((r['study'],r['condition'],r['axis']) for r in rows))
    for study,condition,axis in keys:
        group=[r for r in rows if (r['study'],r['condition'],r['axis'])==(study,condition,axis) and r.get('quality_ok')]
        for metric in numeric:
            vals=np.array([r[metric] for r in group if r.get(metric) is not None],float)
            mean=lo=hi=None
            if len(vals):mean=float(vals.mean())
            if len(vals)>=3:
                boot=rng.choice(vals,(2000,len(vals)),replace=True).mean(axis=1)
                lo,hi=map(float,np.quantile(boot,[.025,.975]))
            summary.append(dict(study=study,condition=condition,axis=axis,metric=metric,n=len(vals),mean=mean,ci_low=lo,ci_high=hi))
    save_json(out/'summary.json',summary)
    with open(out/'summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['study','condition','axis','metric','n','mean','ci_low','ci_high']);w.writeheader();w.writerows(summary)
    units={'apparent_inertia':'kg / kg·m²','axial_effort_rms':'N / N·m','breakaway_peak':'N / N·m',
           'onset_delay_s':'s','rotation_leak_deg_per_m':'deg/m','settling_time_s':'s','core_saturation_fraction':'fraction'}
    selected=list(units)
    for study,axis in sorted(set((s['study'],s['axis']) for s in summary)):
        names=sorted(set(s['condition'] for s in summary if s['study']==study and s['axis']==axis))
        cells={(s['condition'],s['metric']):s for s in summary if s['study']==study and s['axis']==axis}
        width=250+170*len(selected);height=150+45*len(names)
        parts=['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d"><rect width="100%%" height="100%%" fill="white"/>'%(width,height),
               '<g font-family="sans-serif" font-size="12"><text x="10" y="20">%s%s / %s — condition × metric; gray = unavailable; color normalized WITHIN each column</text>'%(synthetic_label,study,axis)]
        for j,m in enumerate(selected):
            x=235+170*j
            parts.append('<text x="%d" y="48">%s</text><text x="%d" y="66">%s</text>'%(x,m,x,units[m]))
            vals=[cells[(n,m)]['mean'] for n in names if cells[(n,m)]['mean'] is not None]
            low=min(vals) if vals else 0; high=max(vals) if vals else 1
            for i,n in enumerate(names):
                s=cells[(n,m)];v=s['mean'];y=85+45*i
                f=0 if v is None else (v-low)/max(high-low,1e-12)
                color='#ddd' if v is None else 'rgb(%d,%d,240)'%(int(235-170*f),int(245-100*f))
                parts.append('<rect x="%d" y="%d" width="165" height="40" fill="%s"/>'%(x,y,color))
                label='NA' if v is None else '%.3g (n=%d)'%(v,s['n'])
                parts.append('<text x="%d" y="%d">%s</text>'%(x+6,y+24,label))
        for i,n in enumerate(names):parts.append('<text x="10" y="%d">%s</text>'%(109+45*i,n))
        (out/('%s_%s_heatmap.svg'%(study,axis))).write_text(''.join(parts)+'</g></svg>')
    return summary


def analyze(a):
    from piperx_teleop.dynamics import ArmDynamics
    payload=json.loads(Path(a.payload).read_text())
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rows=[]
    for trialfile in sorted(Path(a.root).glob('trial_*/trial.json')):
        trialdir=trialfile.parent;info=json.loads(trialfile.read_text());trial=info['trial'];condition=info['condition']
        row=dict(id=trial['id'],condition=condition['id'],study=condition['study'],axis=trial['axis'],
                 participant=trial['participant'],direction=trial['direction'],repeat=trial['repeat'],settings=condition['settings'])
        try:
            meta=json.loads((trialdir/'streams/metadata.json').read_text());config=meta['sensor']
            if not meta['completed']:raise ValueError('incomplete/tripped recording: '+str(meta.get('trip')))
            for key in ('T_tcp_sensor','counts_per_N','counts_per_Nm','wrench_sign'):
                if payload['config'][key]!=config[key]:raise ValueError('payload calibration configuration mismatch: '+key)
            robot=np.genfromtxt(trialdir/'streams/robot.csv',delimiter=',',names=True)
            ft=np.genfromtxt(trialdir/'streams/ft.csv',delimiter=',',names=True)
            dyn=ArmDynamics(tcp_offset=meta['controller']['tcp'],rotor=0)
            data=process_stream(robot,ft,config,payload,dyn)
            row.update(metrics(data,trial['axis'],trial['direction']))
            row['delay_verified']=config.get('delay_verified',False)
            if not row['delay_verified']:
                row['inertia_fit']['timing_warning']='sensor delay not verified; apparent inertia excluded from summary'
                row['apparent_inertia']=None
            if data['guard_active_fraction']>.05 or data['core_saturation_fraction']>.05 or data['ff_saturation_fraction']>.05:
                row['apparent_inertia']=None
                row['inertia_fit']['operating_warning']='guard/cap active >5%; linear target-inertia comparison excluded'
            row['packet_gaps']=meta.get('packet_gaps')
            np.savez(out/(trial['id']+'_processed.npz'),**data)
            line_plot(out/(trial['id']+'_trace.svg'),data['t'],[
                ('TCP linear velocity (m/s)',data['v'][:,:3]),('Human force estimate (N)',data['human'][:,:3]),
                ('TCP angular velocity (rad/s)',data['v'][:,3:]),('Gravity-compensated sensor torque at TCP (N.m)',data['compensated'][:,3:])],trial['id']+' '+condition['id'])
        except Exception as e:row.update(quality_ok=False,error=str(e))
        rows.append(row)
    if not rows:raise ValueError('no trial manifests found')
    save_json(out/'metrics.json',rows)
    fields=sorted(set(k for r in rows for k in r))
    with open(out/'metrics.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:scalar(v) for k,v in r.items()} for r in rows)
    summarize(rows,out)
    print('Wrote',out,'valid acquisition trials',sum(r.get('quality_ok',False) for r in rows),'/',len(rows))


def demo(a):
    from ablation_experiment import conditions
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rows=[]
    # Demonstrate analysis algebra and rendering ONLY; explicitly synthetic.
    for condition in conditions():
        for repeat in range(3):
            t=np.arange(0,12,.01);u=np.clip((t-2)/4,0,1)
            p=.12*(u-np.sin(2*np.pi*u)/(2*np.pi));v=np.gradient(p,.01);acc=np.gradient(v,.01)
            mass=2+1/(1+condition['settings']['balance'])
            force=mass*acc+3*v+.4*np.tanh(v/.005)
            V=np.zeros((len(t),6));V[:,0]=v;W=np.zeros_like(V);W[:,0]=force
            data=dict(t=t,v=V,human=W,compensated=W,displacement=np.column_stack([p,p*0,p*0]),
                      rotation=np.zeros((len(t),3)),rotational_dynamics_valid=True,max_robot_gap=.01,max_ft_gap=.001,
                      sensor_age_p99=.003,core_saturation_fraction=0.,ff_saturation_fraction=0.,trips=0,guard_active_fraction=0.)
            row=metrics(data,'x');row.update(study=condition['study'],condition=condition['id'],repeat=repeat,synthetic=True)
            rows.append(row)
    save_json(out/'SYNTHETIC_metrics.json',rows);summarize(rows,out)
    line_plot(out/'SYNTHETIC_trace.svg',t,[('velocity m/s',v),('force N',force)],'SYNTHETIC DEMO — NOT ROBOT RESULTS')
    print('Synthetic demo only:',out)


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('analyze');p.add_argument('--root',required=True);p.add_argument('--payload',required=True);p.add_argument('--out',required=True);p.set_defaults(func=analyze)
    p=sub.add_parser('demo');p.add_argument('--out',required=True);p.set_defaults(func=demo)
    a=ap.parse_args();a.func(a)


if __name__=='__main__':main()
