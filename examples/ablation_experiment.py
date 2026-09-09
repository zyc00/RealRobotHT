"""Plan/probe/calibrate/run/analyze leader F/T ablations. No automatic trial sweep."""
import argparse
import hashlib
import itertools
import ipaddress
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from ft_measurement import NetFT, load_config, fit_payload


def write_new(path, value):
    with open(path,'x') as f: json.dump(value,f,indent=2)


def conditions(beta=1., vs=.05, damping=(2.,.3), fric_on=.9):
    base=dict(balance=2.,fric_scale=fric_on,balance_breakaway=beta,breakaway_vs=vs,
              damp=list(damping))
    rows=[]
    def add(study, **values):
        c=dict(base,**values)
        # Canonical numeric types make 2 and 2.0 the same physical condition.
        c={k:([float(x) for x in v] if isinstance(v,list) else float(v)) for k,v in c.items()}
        rows.append(dict(id='%s_%02d'%(study,sum(r['study']==study for r in rows)),study=study,
                         settings=c, equivalent_key=json.dumps(c,sort_keys=True),
                         effective_breakaway=c['fric_scale']*c['balance_breakaway']))
    for k in (0,1,2): add('inertia',balance=k)
    for s in (0,.3,.6,.9): add('friction',fric_scale=s)
    for b in (0,1): add('breakaway',balance_breakaway=b)
    for d,r in itertools.product((0,2),(0,.3)):
        add('damping',damp=[d,r],balance=1.5,fric_scale=.7)
    # Decay is irrelevant at beta=0; identify intentional duplicate settings.
    for row in rows:
        c=row['settings'].copy()
        if row['effective_breakaway']==0: c['breakaway_vs']=None;c['balance_breakaway']=0
        row['equivalent_key']=json.dumps(c,sort_keys=True)
    return rows


def plan(a):
    if a.repeats<1 or a.duration<10: raise ValueError('use repeats>=1 and duration>=10 s')
    if not np.isfinite([a.on_decay,*a.on_damping,a.friction_on]).all() or not .005<=a.on_decay<=.3 or not 0<=a.on_damping[0]<=20 or not 0<=a.on_damping[1]<=.3 or not 0<a.friction_on<=.9:
        raise ValueError('invalid on-settings; use controller-supported ranges')
    c=conditions(a.on_breakaway,a.on_decay,tuple(a.on_damping),a.friction_on)
    trials=[]
    for repeat in range(a.repeats):
        block=[]
        for row,axis,sign in itertools.product(c,a.axes,a.signs):
            block.append(dict(condition=row['id'],axis=axis,direction=sign,repeat=repeat,
                              participant=a.participant,duration=a.duration))
        random.Random(a.seed+repeat).shuffle(block)
        trials+=block
    for i,t in enumerate(trials):t['id']='trial_%04d'%i
    write_new(a.out,dict(schema=2,design='reduced_13_rows_damping_k1p5_f0p7',seed=a.seed,conditions=c,trials=trials,
                        on_defaults=dict(beta=a.on_breakaway,vs=a.on_decay,damping=a.on_damping),
                        protocol='0-2 s rest; 2-6 s one directed push/rotation; 6-end release and observe'))
    print('Saved',a.out,':',len(c),'conditions,',len(trials),'trials;',round(sum(t['duration'] for t in trials)/3600,2),'hours recording only.')
    print('No motors commanded. Equal physical settings are marked by equivalent_key.')
    print('WARNING: damping rows include zero damping with high assistance; safety-screen before randomized runs.')


def list_trials(a):
    manifest=json.loads(Path(a.plan).read_text())
    rows={c['id']:c for c in manifest['conditions']}
    n=0
    for trial in manifest['trials']:
        row=rows[trial['condition']]
        if a.study and row['study']!=a.study:continue
        print(trial['id'],row['id'],trial['axis'], '%+d'%trial['direction'],
              'repeat',trial['repeat'], json.dumps(row['settings']))
        n+=1
        if n>=a.limit:break


def probe(a):
    if not np.isfinite(a.seconds) or not 0<a.seconds<=600:raise ValueError('probe duration must be 0..600 s')
    c=load_config(a.config); out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    sensor=NetFT(c,out/'ft.csv').start()
    samples=[];end=time.monotonic()+a.seconds
    try:
        while time.monotonic()<end:
            ts,w,seq=sensor.sample();samples.append(w)
            print('sensor-frame N / N.m:',np.round(w,4), 'age_ms',round((time.monotonic()-ts)*1000,1))
            time.sleep(.2)
    finally:sensor.close()
    write_new(out/'probe.json',dict(config=c,packets=sensor.received,gaps=sensor.gaps,
                                  duplicates=sensor.duplicates,mean=np.mean(samples,axis=0).tolist()))
    print('Transport checked only. Units, sign, mounting transform and overload limits need physical verification.')


def sensor_info(a):
    ip=str(ipaddress.ip_address(a.ip))
    with urllib.request.urlopen('http://'+ip+'/netftapi2.xml',timeout=3) as response:
        xml=response.read(1024*1024)
    values={e.tag.split('}')[-1]:(e.text or '').strip() for e in ET.fromstring(xml).iter()}
    keys=('netip','cfgcalsn','scfgfu','scfgtu','cfgcpf','cfgcpt','runrate')
    print(json.dumps({k:values.get(k) for k in keys},indent=2))
    c=json.loads((Path(__file__).parents[1]/'configs/ft_experiment.example.json').read_text())
    c['ip']=ip;c['sensor_serial']=values.get('cfgcalsn','unknown')
    if values.get('scfgfu')=='N' and values.get('scfgtu') in ('Nm','N-m','N.m'):
        c['counts_per_N']=float(values['cfgcpf']);c['counts_per_Nm']=float(values['cfgcpt'])
        c['units_verified']=c['counts_per_N']>0 and c['counts_per_Nm']>0
    else:print('Non-SI or unknown active units: explicit conversion to counts/N and counts/N.m required.')
    write_new(a.out,c)
    with open(str(a.out)+'.xml','xb') as f:f.write(xml)
    print('Saved draft',a.out,'; geometry/sign and timing still require verification. No hardware settings changed.')


def calibrate(a):
    if a.poses<6:raise ValueError('at least six diverse stationary orientations required')
    from piperx_teleop.arm import PiperArm
    from piperx_teleop.dynamics import ArmDynamics
    from position_follower import fresh_q
    c=load_config(a.config,geometry=True);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    dyn=ArmDynamics(tcp_offset=a.tcp,rotor=0)
    Tts=np.asarray(c['T_tcp_sensor'])
    arm=PiperArm(a.can).connect(require_control=False)
    sensor=None;rotations=[];wrenches=[];poses=[]
    try:
        sensor=NetFT(c,out/'ft.csv').start()
        print('READ-ONLY calibration. This script never enables, releases or moves the arm.')
        print('Use your safe positioning procedure between poses. Hold rigidly, then remove all hand/contact forces for each sample.')
        for i in range(a.poses):
            input('Pose %d/%d: vary orientation substantially; stationary and hands clear, then ENTER: '%(i+1,a.poses))
            qs=[];ws=[];last=None;end=time.monotonic()+2
            while time.monotonic()<end:
                q=fresh_q(arm,time.time());ts,w,seq=sensor.sample()
                if seq!=last:qs.append(q);ws.append(w*c['wrench_sign']);last=seq
                time.sleep(.01)
            if len(qs)<50 or np.max(np.ptp(qs,axis=0))>np.radians(.3):
                raise RuntimeError('calibration pose moved or insufficient unique samples; collected data saved')
            q=np.mean(qs,axis=0);poses.append(q.tolist())
            rotations.append((dyn.tcp_pose(q)@Tts)[:3,:3]);wrenches.append(np.mean(ws,axis=0))
            write_new(out/('pose_%02d.json'%i),dict(q=poses[-1],rotation=rotations[-1].tolist(),wrench=wrenches[-1].tolist()))
        result=fit_payload(rotations,wrenches)
        residual=[]
        from ft_measurement import gravity_design
        for i in range(len(rotations)):
            idx=[j for j in range(len(rotations)) if j!=i]
            fit=fit_payload(np.asarray(rotations)[idx],np.asarray(wrenches)[idx])
            theta=np.r_[fit['mass_kg'],fit['mass_kg']*np.asarray(fit['com_sensor_m']),fit['bias']]
            residual.append(wrenches[i]-gravity_design(rotations[i])@theta)
        result['leave_one_pose_out_rmse']=np.sqrt(np.mean(np.asarray(residual)**2,axis=0)).tolist()
        result.update(config=c,source='static_orientation_fit',inertia_sensor_com_kgm2=None,
                      note='Distal body below sensor only; rotational inertia must come from CAD/separate measurement.')
        write_new(out/'payload.json',result);print(json.dumps(result,indent=2))
    finally:
        try:
            if sensor:sensor.close()
        finally:arm.close()


def run(a):
    manifest=json.loads(Path(a.plan).read_text())
    trial=next(t for t in manifest['trials'] if t['id']==a.trial)
    condition=next(c for c in manifest['conditions'] if c['id']==trial['condition'])
    load_config(a.config,geometry=True)
    out=Path(a.root)/trial['id'];out.mkdir(parents=True,exist_ok=False)
    provenance={}
    for name in ('tool','gravity','friction','config'):
        p=Path(getattr(a,name));provenance[name]=dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    provenance['code']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__).with_name(n) for n in ('b601_drag.py','joint_breakaway.py','b601_reference_balance.py','ft_measurement.py','ablation_metrics.py')]}
    write_new(out/'trial.json',dict(trial=trial,condition=condition,provenance=provenance))
    cmd=[sys.executable,str(Path(__file__).with_name('drag_mode.py')),'--can',a.can,
         '--tool',a.tool,'--gravity',a.gravity,'--friction',a.friction,
         '--measurement-config',a.config,'--measurement-dir',str(out/'streams'),
         '--log',str(out/'controller.npz'),'--duration',str(trial['duration']),'--serve']
    for key,value in condition['settings'].items():
        cmd+=['--'+key.replace('_','-')]+[str(x) for x in (value if isinstance(value,list) else [value])]
    print(json.dumps(dict(trial=trial,settings=condition['settings']),indent=2))
    print('Axes fixed at initial TCP pose: x/y/z translation; roll=tool z, pitch=tool y, yaw=tool x.')
    print('Use SAME initial pose, grip point, distal load and approximate speed/profile across conditions.')
    print('0-2 s: hands clear/rest. 2-6 s: ONE smooth directed push with acceleration/deceleration. 6-end: release, observe.')
    print('Translation trials: try to keep orientation fixed. Rotational trials: minimize TCP translation.')
    print('HIGH ASSIST CONDITIONS MAY BE UNSTABLE. No follower is enabled by this runner. E-stop accessible.')
    if input('Type RUN to launch this one trial: ').strip()!='RUN':raise RuntimeError('cancelled')
    if any(d==0 for d in condition['settings']['damp']):
        print('WARNING: zero damping component with assistance can cause drift/oscillation. This is not certified passive.')
        if input('Only after a supported safety check, type LOW DAMPING to proceed: ').strip()!='LOW DAMPING':
            raise RuntimeError('zero-damping trial cancelled')
    result=subprocess.run(cmd)
    write_new(out/'exit.json',dict(returncode=result.returncode))
    if result.returncode:raise SystemExit(result.returncode)


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--participant',default='operator01');p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--axes',nargs='+',choices=['x','y','z','roll','pitch','yaw'],default=['x','y','z'])
    p.add_argument('--signs',nargs='+',type=int,choices=[-1,1],default=[-1,1]);p.add_argument('--duration',type=float,default=12)
    p.add_argument('--on-breakaway',type=float,choices=[.5,1.],default=1.)
    p.add_argument('--on-decay',type=float,default=.05);p.add_argument('--on-damping',nargs=2,type=float,default=[2.,.3])
    p.add_argument('--friction-on',type=float,default=.9);p.set_defaults(func=plan)
    p=sub.add_parser('list');p.add_argument('--plan',required=True)
    p.add_argument('--study',choices=['inertia','friction','breakaway','damping']);p.add_argument('--limit',type=int,default=20);p.set_defaults(func=list_trials)
    p=sub.add_parser('probe');p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seconds',type=float,default=5);p.set_defaults(func=probe)
    p=sub.add_parser('sensor-info');p.add_argument('--ip',required=True);p.add_argument('--out',required=True);p.set_defaults(func=sensor_info)
    p=sub.add_parser('calibrate');p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--can',default='can0');p.add_argument('--poses',type=int,default=12);p.add_argument('--tcp',type=float,default=.19);p.set_defaults(func=calibrate)
    p=sub.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--trial',required=True)
    p.add_argument('--config',required=True);p.add_argument('--root',required=True);p.add_argument('--can',default='can0')
    for name,default in [('tool','data/tool_body.npz'),('gravity','data/gravity_cal.npz'),('friction','data/friction_model.npz')]:p.add_argument('--'+name,default=default)
    p.set_defaults(func=run)
    a=ap.parse_args();a.func(a)


if __name__=='__main__':main()
