import json
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from ablation_experiment import conditions
from ablation_metrics import integral_fit, metrics, process_stream
from ft_measurement import fit_payload, gravity_design, NetFT, load_config


class ExperimentTests(unittest.TestCase):
    def test_counts_and_breakaway_coupling(self):
        rows=conditions()
        self.assertEqual([sum(r['study']==s for r in rows) for s in ('inertia','friction','breakaway','damping')],[3,4,2,4])
        self.assertTrue(all(r['effective_breakaway']==0 for r in rows if r['settings']['fric_scale']==0))
        b=[r['equivalent_key'] for r in rows if r['study']=='breakaway']
        self.assertEqual(len(set(b)),2)
        self.assertEqual(len(set(r['equivalent_key'] for r in rows)),11)
        self.assertTrue(all(r['settings']['breakaway_vs']==.05 for r in rows))
        self.assertEqual([r['settings']['fric_scale'] for r in rows if r['study']=='friction'],[0,.3,.6,.9])
        self.assertTrue(all(r['settings']['balance']==2 for r in rows if r['study'] in ('friction','breakaway')))
        self.assertTrue(all(r['settings']['balance']==1.5 and r['settings']['fric_scale']==.7
                            for r in rows if r['study']=='damping'))
        self.assertEqual([r['settings']['damp'] for r in rows if r['study']=='damping'],[[0,0],[0,.3],[2,0],[2,.3]])

    def test_payload_fit_and_insufficient_diversity(self):
        R=Rotation.random(20,random_state=3).as_matrix()
        theta=np.r_[.25,.25*np.array([.01,-.02,.05]),[.1,-.2,.3,.01,.02,-.01]]
        W=np.array([gravity_design(r)@theta for r in R])
        result=fit_payload(R,W)
        self.assertAlmostEqual(result['mass_kg'],.25)
        np.testing.assert_allclose(result['com_sensor_m'],[.01,-.02,.05],atol=1e-10)
        with self.assertRaises(ValueError):fit_payload(np.tile(np.eye(3),(20,1,1)),W)

    def test_integral_fit_known_mass_and_no_excitation(self):
        t=np.arange(0,12,.002)
        v=.1*np.sin(2*t)+.04*np.sin(5*t);acc=.2*np.cos(2*t)+.2*np.cos(5*t)
        F=2.5*acc+4*v+.7*np.tanh(v/.005)+.1
        result=integral_fit(t,v,F)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['apparent_inertia'],2.5,places=3)
        self.assertFalse(integral_fit(t,np.zeros_like(t),F)['valid'])

    def test_template_fails_closed(self):
        with self.assertRaises(ValueError):load_config(Path(__file__).parents[1]/'configs/ft_experiment.example.json')

    def test_rdt_status_gap_duplicates_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(ip='127.0.0.1',counts_per_N=100,counts_per_Nm=1000)
            n=NetFT(config,Path(tmp)/'ft.csv')
            packets=[struct.pack('>IIIiiiiii',seq,seq,status,100,200,-300,1000,-2000,3000)
                     for seq,status in [(1,0),(1,0),(3,0),(4,8)]]
            class FakeSocket:
                def recv(self,_):return packets.pop(0)
            n.sock=FakeSocket();n._loop()
            self.assertIn('status fault',str(n.error));self.assertEqual(n.gaps,1);self.assertEqual(n.duplicates,1)
            np.testing.assert_allclose(n.latest[1],[1,2,-3,1,-2,3])
            n.sink.close()
            rows=np.genfromtxt(Path(tmp)/'ft.csv',delimiter=',',names=True)
            self.assertEqual(len(rows),3);self.assertEqual(rows['status'][-1],8)

    def test_gravity_and_wrench_origin_transform(self):
        N=300;t=np.arange(N)*.01
        fields=['mono','sensor_age','core_saturated','ff_saturated','trips']+['q%d'%j for j in range(1,7)]+['safety_delta%d'%j for j in range(1,7)]
        robot=np.zeros(N,dtype=[(x,float) for x in fields]);robot['mono']=t
        fields=['receive_mono','status']+['count_'+x for x in ('fx','fy','fz','tx','ty','tz')]
        ft=np.zeros(N,dtype=[(x,float) for x in fields]);ft['receive_mono']=t
        ft['count_fx']=1;ft['count_fz']=-.2*9.80665
        T=np.eye(4);T[2,3]=.1
        config=dict(counts_per_N=1,counts_per_Nm=1,wrench_sign=1,T_tcp_sensor=T.tolist())
        payload=dict(mass_kg=.2,com_sensor_m=[0,0,0],bias=[0]*6)
        class Dyn:
            def tcp_pose(self,q):return np.eye(4)
        data=process_stream(robot,ft,config,payload,Dyn())
        np.testing.assert_allclose(data['human'][:,0],1,atol=1e-10)
        np.testing.assert_allclose(data['human'][:,4],.1,atol=1e-10)
        np.testing.assert_allclose(data['human'][:,2],0,atol=1e-10)

    def test_saved_trial_to_figures_end_to_end(self):
        from analyze_ablations import analyze
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);trialdir=root/'runs/trial_0000';streams=trialdir/'streams';streams.mkdir(parents=True)
            row=conditions()[0]
            trial=dict(id='trial_0000',condition=row['id'],axis='x',direction=1,repeat=0,participant='synthetic')
            (trialdir/'trial.json').write_text(json.dumps(dict(trial=trial,condition=row)))
            config=dict(T_tcp_sensor=np.eye(4).tolist(),counts_per_N=1000,counts_per_Nm=1000,wrench_sign=1,delay_verified=True)
            payload=dict(config=config,mass_kg=.2,com_sensor_m=[0,0,0],bias=[0]*6,inertia_sensor_com_kgm2=np.eye(3).tolist())
            (root/'payload.json').write_text(json.dumps(payload))
            (streams/'metadata.json').write_text(json.dumps(dict(completed=True,sensor=config,controller=dict(tcp=.19),packet_gaps=0)))
            t=np.arange(0,12,.01);u=np.clip((t-2)/4,0,1);p=.12*(u-np.sin(2*np.pi*u)/(2*np.pi))
            v=np.gradient(p,.01);acc=np.gradient(v,.01);F=2.5*acc+3*v+.4*np.tanh(v/.005)
            names=['mono','sensor_age','core_saturated','ff_saturated','trips']+['q%d'%j for j in range(1,7)]+['safety_delta%d'%j for j in range(1,7)]
            robot=np.zeros((len(t),len(names)));robot[:,0]=t;robot[:,5]=p
            np.savetxt(streams/'robot.csv',robot,delimiter=',',header=','.join(names),comments='')
            names=['receive_mono','status']+['count_'+x for x in ('fx','fy','fz','tx','ty','tz')]
            ft=np.zeros((len(t),len(names)));ft[:,0]=t;ft[:,2]=(F-.2*acc)*1000;ft[:,4]=-.2*9.80665*1000
            np.savetxt(streams/'ft.csv',ft,delimiter=',',header=','.join(names),comments='')
            class Dyn:
                def __init__(self,**kwargs):pass
                def tcp_pose(self,q):
                    T=np.eye(4);T[0,3]=q[0];return T
            with patch('piperx_teleop.dynamics.ArmDynamics',Dyn):
                analyze(SimpleNamespace(root=root/'runs',payload=root/'payload.json',out=root/'plots'))
            result=json.loads((root/'plots/metrics.json').read_text())[0]
            self.assertTrue(result['quality_ok'])
            self.assertAlmostEqual(result['apparent_inertia'],2.5,delta=.05)
            self.assertTrue((root/'plots/inertia_x_heatmap.svg').exists())
            self.assertTrue((root/'plots/trial_0000_trace.svg').exists())


if __name__=='__main__':unittest.main()
