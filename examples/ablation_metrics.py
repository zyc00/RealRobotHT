"""Offline measurements: fixed initial-TCP frame, explicit distal dynamics.

No claim that measured free-space effort equals unassisted joint friction.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from scipy.spatial.transform import Rotation

AXIS={'x':0,'y':1,'z':2,'roll':5,'pitch':4,'yaw':3}


def smooth(x, dt, derivative=0):
    n=min(len(x)-(1-len(x)%2),max(5,int(.11/dt)//2*2+1))
    if n<5:raise ValueError('too few samples')
    return savgol_filter(x,n,3,deriv=derivative,delta=dt,axis=0)


def angular_velocity(R,dt):
    increments=Rotation.from_matrix(R[1:]@np.transpose(R[:-1],(0,2,1))).as_rotvec()/dt
    return np.vstack([increments[0],(increments[:-1]+increments[1:])/2,increments[-1]])


def process_stream(robot,ft,config,payload,dyn,dt=.01):
    rt=robot['mono'];st=ft['receive_mono']-float(config.get('sensor_delay_s',0))
    if len(rt)<100 or len(st)<100:raise ValueError('insufficient robot/F/T data')
    if np.any(np.diff(rt)<=0) or np.any(np.diff(st)<0):raise ValueError('nonmonotonic timestamps')
    if np.any(ft['status']!=0):raise ValueError('sensor fault in raw stream')
    start=max(rt[0],st[0]);end=min(rt[-1],st[-1]);t=np.arange(start,end,dt)
    if len(t)<100:raise ValueError('insufficient overlapping robot/F/T data')
    q=np.column_stack([np.interp(t,rt,robot['q%d'%j]) for j in range(1,7)])
    pose=np.array([dyn.tcp_pose(x).copy() for x in q]);R=pose[:,:3,:3];p=pose[:,:3,3]
    Tts=np.asarray(config['T_tcp_sensor']);sensor=pose@Tts
    Rs=sensor[:,:3,:3];ps=sensor[:,:3,3];R0=R[0]
    scales=[config['counts_per_N']]*3+[config['counts_per_Nm']]*3
    # Average high-rate measurements in each host-time bin, rather than pick
    # one sample and alias all high-rate noise into the robot sampling rate.
    raw_columns=[]
    left=np.searchsorted(st,t-dt/2);right=np.searchsorted(st,t+dt/2)
    for key,scale in zip(('fx','fy','fz','tx','ty','tz'),scales):
        values=ft['count_'+key]/scale;cumulative=np.r_[0.,np.cumsum(values)]
        binned=(cumulative[right]-cumulative[left])/np.maximum(right-left,1)
        binned=np.where(right>left,binned,np.interp(t,st,values))
        raw_columns.append(binned)
    raw=np.column_stack(raw_columns)
    raw=raw*config['wrench_sign']-np.asarray(payload['bias'])
    m=payload['mass_kg'];c=np.asarray(payload['com_sensor_m'])
    gs=np.einsum('nji,j->ni',Rs,np.array([0,0,-9.80665]))
    wg=np.column_stack([m*gs,np.cross(np.broadcast_to(c,gs.shape),m*gs)])
    local=raw-wg
    force=np.einsum('nij,nj->ni',Rs,local[:,:3]);torque=np.einsum('nij,nj->ni',Rs,local[:,3:])
    torque+=np.cross(ps-p,force)
    # W_comp and distal inertia use the SAME TCP origin and task axes.
    w=angular_velocity(Rs,dt);alpha=smooth(w,dt,1)
    r=np.einsum('nij,j->ni',Rs,c)
    acom=smooth(ps,dt,2)+np.cross(alpha,r)+np.cross(w,np.cross(w,r))
    fi=m*acom;ti=np.cross(r,fi)
    inertia=payload.get('inertia_sensor_com_kgm2')
    if inertia is not None:
        I=np.asarray(inertia,float)
        if I.shape!=(3,3) or not np.allclose(I,I.T) or np.min(np.linalg.eigvalsh(I))<=0:
            raise ValueError('invalid distal rotational inertia')
        Ib=Rs@I@np.transpose(Rs,(0,2,1))
        ti+=np.einsum('nij,nj->ni',Ib,alpha)+np.cross(w,np.einsum('nij,nj->ni',Ib,w))
    ti+=np.cross(ps-p,fi)
    compensated=np.column_stack([force@R0,torque@R0])
    human=np.column_stack([(force+fi)@R0,(torque+ti)@R0])
    v=np.column_stack([smooth(p,dt,1)@R0,smooth(angular_velocity(R,dt),dt)@R0])
    displacement=(p-p[0])@R0
    rotation=Rotation.from_matrix(R0.T@R).as_rotvec()
    return dict(t=t-t[0],q=q,v=v,displacement=displacement,rotation=rotation,
                compensated=compensated,human=human,rotational_dynamics_valid=inertia is not None,
                max_robot_gap=float(np.max(np.diff(rt))),max_ft_gap=float(np.max(np.diff(st))),
                sensor_age_p99=float(np.quantile(robot['sensor_age'],.99)),
                core_saturation_fraction=float(np.mean(robot['core_saturated'])),
                ff_saturation_fraction=float(np.mean(robot['ff_saturated'])),
                trips=int(np.max(robot['trips'])),
                guard_active_fraction=float(np.mean(np.any(np.column_stack([robot['safety_delta%d'%j] for j in range(1,7)])!=0,axis=1))))


def integral_fit(t,v,force,window=.25):
    """Non-overlapping windows: integral F = M dv + B dx + C int tanh(v/v0) + offset dt."""
    dt=float(np.median(np.diff(t)));step=max(5,int(window/dt));X=[];y=[]
    for i in range(0,len(t)-step,step):
        s=slice(i,i+step+1);ts=t[s]
        X.append([v[i+step]-v[i],trapezoid(v[s],ts),trapezoid(np.tanh(v[s]/.005),ts),ts[-1]-ts[0]])
        y.append(trapezoid(force[s],ts))
    X=np.asarray(X);y=np.asarray(y)
    if len(y)<12: return dict(valid=False,reason='fewer than 12 independent windows')
    scales=np.linalg.norm(X,axis=0)
    if np.any(scales<1e-8):return dict(valid=False,reason='insufficient excitation')
    A=X/scales;condition=float(np.linalg.cond(A))
    if np.linalg.matrix_rank(A)<4 or condition>1000:return dict(valid=False,reason='ill-conditioned fit',condition=condition)
    theta=np.linalg.lstsq(A,y,rcond=None)[0]/scales
    r2=1-np.sum((X@theta-y)**2)/max(np.sum((y-y.mean())**2),1e-12)
    valid=bool(theta[0]>0 and r2>.5)
    return dict(valid=valid,reason='' if valid else 'nonpositive inertia or poor fit',
                apparent_inertia=float(theta[0]),damping=float(theta[1]),coulomb=float(theta[2]),
                bias=float(theta[3]),r2=float(r2),condition=condition,windows=len(y))


def sustained(mask,n):
    idx=np.flatnonzero(np.convolve(np.asarray(mask,int),np.ones(n,int),'valid')==n)
    return None if not len(idx) else int(idx[0])


def metrics(data,axis,direction=1):
    t=data['t'];v=data['v'];W=data['human'];comp=data['compensated'];j=AXIS[axis]
    dt=float(np.median(np.diff(t)));translation=j<3
    active=(t>=2)&(t<=6);rest=t<1.5;release=t>=6
    if not active.any() or t[-1]<9:raise ValueError('trial too short for push/release protocol')
    usable=bool(data['max_robot_gap']<.05 and data['max_ft_gap']<.05 and data['trips']==0)
    effort=W[:,j] if translation or data['rotational_dynamics_valid'] else comp[:,j]
    path=float(trapezoid(abs(v[active,j]),t[active]))
    results=dict(axis=axis,direction=direction,quality_ok=usable,path=path,
                 force_rms_N=float(np.sqrt(np.mean(np.sum(W[active,:3]**2,axis=1)))),
                 torque_quasistatic_rms_Nm=float(np.sqrt(np.mean(np.sum(comp[active,3:]**2,axis=1)))),
                 axial_effort_rms=float(np.sqrt(np.mean(effort[active]**2))),
                 peak_axial_effort=float(np.max(abs(effort[active]))),
                 intended_direction_fraction=float(np.mean(v[active,j]*direction>0)),
                 rotational_dynamics_valid=data['rotational_dynamics_valid'])
    for name in ('core_saturation_fraction','ff_saturation_fraction','guard_active_fraction','max_robot_gap','max_ft_gap','sensor_age_p99','trips'):
        results[name]=data[name]
    if translation and path>.03:
        results['rotation_leak_deg_per_m']=float(np.degrees(trapezoid(np.linalg.norm(v[active,3:],axis=1),t[active]))/path)
        results['orientation_peak_deg']=float(np.degrees(np.max(np.linalg.norm(data['rotation'][active],axis=1))))
    else:results['rotation_leak_deg_per_m']=None
    speed_threshold=.005 if translation else .03
    onset=sustained((t>=2)&(t<6)&(v[:,j]*direction>speed_threshold),max(2,int(.15/dt)))
    baseline=float(np.median(effort[rest]));noise=float(np.std(effort[rest]))
    force_threshold=max(5*noise,.5 if translation else .03)
    push=sustained((t>=2)&(t<6)&(abs(effort-baseline)>force_threshold),max(2,int(.05/dt)))
    results.update(breakaway_peak=None,onset_delay_s=None)
    if onset is not None and push is not None and onset>=push:
        results['breakaway_peak']=float(np.max(abs(effort[push:onset+1]-baseline)))
        results['onset_delay_s']=float(t[onset]-t[push])
    # Release metrics are valid only if force confirms that the hand was removed.
    release_torque=W[:,3:] if data['rotational_dynamics_valid'] else comp[:,3:]
    released=(t>=6.5)&(np.linalg.norm(W[:,:3],axis=1)<2)&(np.linalg.norm(release_torque,axis=1)<.15)
    release_ok=bool(np.mean(released[t>=6.5])>.8)
    results['release_confirmed']=release_ok
    results['post_release_translation_m']=float(trapezoid(np.linalg.norm(v[release,:3],axis=1),t[release])) if release_ok else None
    results['post_release_rotation_deg']=float(np.degrees(trapezoid(np.linalg.norm(v[release,3:],axis=1),t[release]))) if release_ok else None
    settled=sustained(released&(np.linalg.norm(v[:,:3],axis=1)<.005)&(np.linalg.norm(v[:,3:],axis=1)<.03),max(2,int(.5/dt)))
    results['settling_time_s']=None if settled is None or not release_ok else float(t[settled]-6)
    # Translation work remains meaningful without distal rotational inertia.
    pt=np.sum(W[:,:3]*v[:,:3],axis=1)
    results['translation_work_J']=float(trapezoid(pt,t))
    results['translation_positive_work_J']=float(trapezoid(np.maximum(pt,0),t))
    if data['rotational_dynamics_valid']:
        power=np.sum(W*v,axis=1)
        results['interaction_work_J']=float(trapezoid(power,t))
        results['negative_interaction_work_J']=float(trapezoid(np.minimum(power,0),t))
    fit=integral_fit(t,v[:,j],effort) if translation or data['rotational_dynamics_valid'] else dict(valid=False,reason='distal rotational inertia missing')
    # Axis isolation and timing calibration are additional publication gates.
    transverse=np.delete(v[:,:3] if translation else v[:,3:],j if translation else j-3,axis=1)
    purity=float(trapezoid(np.linalg.norm(transverse[active],axis=1),t[active])/max(path,1e-9))
    results['off_axis_path_ratio']=purity
    fit['valid']=bool(fit['valid'] and usable and purity<.3 and path>(.03 if translation else .1))
    results['inertia_fit']=fit
    results['apparent_inertia']=fit.get('apparent_inertia') if fit['valid'] else None
    return results
