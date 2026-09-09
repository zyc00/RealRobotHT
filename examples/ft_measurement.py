"""ATI Net F/T acquisition and explicit sensor-frame/payload calibration.

No sensor tare, configuration write, arm enable, or arm motion is performed here.
Timestamps are host monotonic RECEIVE times, not device acquisition timestamps.
"""
import csv
import json
from pathlib import Path
import queue
import socket
import struct
import threading
import time

import numpy as np


def load_config(path, geometry=False):
    c = json.loads(Path(path).read_text())
    for key in ('counts_per_N', 'counts_per_Nm'):
        if c.get(key) is None or not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('verify and fill ' + key + ' from the active ATI calibration')
    if not c.get('ip') or c.get('units_verified') is not True:
        raise ValueError('fill sensor IP and explicitly verify N/N.m conversion')
    if geometry:
        T = np.asarray(c.get('T_tcp_sensor'), float)
        if T.shape != (4,4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1]):
            raise ValueError('T_tcp_sensor must be a measured 4x4 sensor-to-TCP transform')
        if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(T[:3,:3]),1):
            raise ValueError('sensor rotation must be proper orthonormal')
        if c.get('wrench_sign') not in (-1,1) or c.get('geometry_verified') is not True:
            raise ValueError('verify sensor transform and external-wrench sign before metrics')
    return c


class CsvSink:
    """Bounded asynchronous writer; full queue is a fault, not silent data loss."""
    def __init__(self, path, header):
        self.file = open(path, 'x', newline='')
        self.writer = csv.writer(self.file); self.writer.writerow(header); self.file.flush()
        self.queue = queue.Queue(20000)
        self.error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def put(self, row):
        if self.error: raise RuntimeError('measurement writer failed: '+str(self.error))
        try: self.queue.put_nowait(row)
        except queue.Full: raise RuntimeError('measurement writer queue full')

    def _loop(self):
        try:
            count = 0
            while True:
                row = self.queue.get()
                if row is None: break
                self.writer.writerow(row); count += 1
                if count % 100 == 0: self.file.flush()
        except Exception as e: self.error = e
        finally: self.file.close()

    def close(self):
        if self.thread.is_alive():
            self.queue.put(None, timeout=2)
            self.thread.join(timeout=5)
        if self.thread.is_alive() or self.error:
            raise RuntimeError('measurement writer did not finish: '+str(self.error))


class NetFT:
    record = struct.Struct('>IIIiiiiii')
    def __init__(self, config, path):
        self.config = config
        self.sink = CsvSink(path, ['receive_mono','receive_wall','rdt_seq','ft_seq','status']+
                            ['count_'+x for x in ('fx','fy','fz','tx','ty','tz')]+['gap'])
        self.latest = None; self.error = None; self.received = self.gaps = self.duplicates = 0
        self.lock = threading.Lock(); self.stop_event = threading.Event()
        self.sock = None; self.thread = None

    def start(self):
        try:
            self.sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            self.sock.connect((self.config['ip'],self.config.get('port',49152)))
            self.sock.settimeout(.1)
            self.sock.send(struct.pack('>HHI',0x1234,2,0))
            self.thread = threading.Thread(target=self._loop,daemon=True); self.thread.start()
            end = time.monotonic()+2
            while self.latest is None and self.error is None and time.monotonic()<end: time.sleep(.01)
            self.sample()
        except Exception:
            self.close(); raise
        return self

    def _loop(self):
        previous = None; previous_ft = None
        try:
            while not self.stop_event.is_set():
                try: packet = self.sock.recv(65536)
                except socket.timeout: continue
                mono, wall = time.monotonic(), time.time()
                if len(packet) != self.record.size: raise RuntimeError('invalid RDT record length')
                rdt, ft, status, *counts = self.record.unpack(packet)
                delta = 1 if previous is None else (rdt-previous)&0xffffffff
                if delta == 0:
                    self.duplicates += 1; continue
                if delta > 0x7fffffff: raise RuntimeError('RDT sequence went backward/reset')
                if previous_ft is not None and ((ft-previous_ft)&0xffffffff) > 0x7fffffff:
                    raise RuntimeError('F/T sequence went backward/reset')
                gap = delta-1; self.gaps += gap; self.received += 1
                self.sink.put([mono,wall,rdt,ft,status,*counts,gap])
                if status: raise RuntimeError('ATI status fault 0x%08x' % status)
                # Repeated device measurement does not refresh freshness time.
                if ft != previous_ft:
                    scale = np.array([self.config['counts_per_N']]*3+[self.config['counts_per_Nm']]*3)
                    with self.lock: self.latest = (mono,np.asarray(counts)/scale,ft)
                previous, previous_ft = rdt, ft
        except Exception as e: self.error = e

    def sample(self, max_age=.15):
        if self.error: raise RuntimeError('F/T acquisition failed: '+str(self.error))
        with self.lock: result = self.latest
        if result is None or time.monotonic()-result[0]>max_age:
            raise RuntimeError('no fresh F/T measurement; check IP, Ethernet, RDT and power')
        return result

    def close(self):
        self.stop_event.set()
        if self.sock:
            try: self.sock.send(struct.pack('>HHI',0x1234,0,0))
            except OSError: pass
        if self.thread: self.thread.join(timeout=1)
        if self.sock: self.sock.close(); self.sock=None
        self.sink.close()


class Capture:
    def __init__(self, config_path, directory, arm, configuration):
        self.directory = Path(directory); self.directory.mkdir(parents=True,exist_ok=False)
        self.arm = arm; self.config=load_config(config_path,geometry=True)
        self.meta=dict(sensor=self.config, controller=configuration, timestamp='host_monotonic_receive',
                       start_wall=time.time(),completed=False)
        (self.directory/'metadata.json').write_text(json.dumps(self.meta,indent=2))
        names=['mono','wall','joint_feedback_wall','control_t','sensor_age']
        for prefix in ('q','velocity','gravity','torque_ff','residual','breakaway','safety_delta'):
            names += [prefix+str(j+1) for j in range(6)]
        names += ['alpha','trips','core_saturated','ff_saturated']
        self.robot=CsvSink(self.directory/'robot.csv',names)
        self.sensor=NetFT(self.config,self.directory/'ft.csv')
        self.phase = None
        try: self.sensor.start()
        except Exception:
            self.robot.close(); raise

    def record(self, s, law):
        mono=time.monotonic(); ft=self.sensor.sample()
        phase = 'REST — hands clear' if s.t<2 else ('MOVE — one directed push/rotation' if s.t<6 else 'RELEASE — hands clear, observe')
        if phase != self.phase:
            print('\aEXPERIMENT %.2f s: %s' % (s.t,phase),flush=True)
            self.phase=phase
        d=law.telem
        row=[mono,time.time(),self.arm.obs_time(),s.t,mono-ft[0]]
        for key in ('q','velocity','gravity','torque_ff','residual','breakaway_torque','safety_delta'):
            row += d[key]
        row += [d['alpha'],d['trips'],int(any(d['core_saturated'])),int(any(d['feedforward_saturated']))]
        self.robot.put(row)

    def close(self, trip=None):
        try: self.sensor.close()
        finally: self.robot.close()
        self.meta.update(completed=trip is None,trip=trip,packets=self.sensor.received,
                         packet_gaps=self.sensor.gaps,duplicates=self.sensor.duplicates,end_wall=time.time())
        (self.directory/'metadata.json').write_text(json.dumps(self.meta,indent=2))


def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])


def gravity_design(R_base_sensor):
    """Parameters [mass, first moment xyz, constant bias wrench six]."""
    gs=R_base_sensor.T@np.array([0,0,-9.80665])
    A=np.zeros((6,10)); A[:3,0]=gs; A[3:,1:4]=-skew(gs); A[:,4:]=np.eye(6)
    return A


def fit_payload(rotations, wrenches):
    A=np.concatenate([gravity_design(R) for R in rotations]); y=np.asarray(wrenches).reshape(-1)
    scale=np.linalg.norm(A,axis=0); scaled=A/np.maximum(scale,1e-12)
    if np.linalg.matrix_rank(scaled)<10 or np.linalg.cond(scaled)>100:
        raise ValueError('insufficient orientation diversity for mass/COM/bias fit')
    theta=np.linalg.lstsq(A,y,rcond=None)[0]
    if not .001<theta[0]<5 or np.linalg.norm(theta[1:4]/theta[0])>.5:
        raise ValueError('implausible distal mass/COM; check wrench sign, units and frames')
    residual=(y-A@theta).reshape(-1,6)
    return dict(mass_kg=float(theta[0]),com_sensor_m=(theta[1:4]/theta[0]).tolist(),
                bias=theta[4:].tolist(),rms=residual.std(axis=0).tolist(),
                condition=float(np.linalg.cond(scaled)),poses=len(rotations))
