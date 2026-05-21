######################################################
#
# PyRAI2MD 2 module for NN/GAFF calculations
#
######################################################

import copy
import numpy as np

from PyRAI2MD.Machine_Learning.model_helper import DummyModel
from PyRAI2MD.Quantum_Chemistry.qc_gaff import Gaff

DNN = None


def _load_dnn():
    global DNN
    if DNN is None:
        try:
            from PyRAI2MD.Machine_Learning.model_NN import DNN as LoadedDNN
            DNN = LoadedDNN
        except Exception:
            DNN = DummyModel
    return DNN


class NNGaff:
    """Subtractive NN + GAFF interface.

    E = E_GAFF(active) - E_GAFF(highlevel) + E_NN(highlevel)
    """

    def __init__(self, keywords=None, job_id_1=None, job_id_2=None):
        dnn = _load_dnn()
        self.qm_high = dnn(keywords=keywords, job_id=job_id_1, runtype='qm_high')
        self.mm_high = Gaff(keywords=keywords, job_id=job_id_2, runtype='qm_high')
        self.mm_all = Gaff(keywords=keywords, job_id=job_id_2, runtype='qm_high_mid_low')
        self.nprocs = keywords['control']['ms_ncpu']

    def train(self):
        return self

    def load(self):
        self.qm_high.load()
        self.mm_high.load()
        self.mm_all.load()
        return self

    def appendix(self, _):
        return self

    def evaluate(self, traj):
        traj_mm_all = self.mm_all.evaluate(copy.deepcopy(traj))
        traj_mm_high = self.mm_high.evaluate(copy.deepcopy(traj))
        traj_qm_high = self.qm_high.evaluate(copy.deepcopy(traj))

        nstate = len(traj_qm_high.energy)
        natom = traj.natom
        highlevel = traj.highlevel

        energy_mm_all = np.repeat(traj_mm_all.energy, nstate)
        energy_mm_high = np.repeat(traj_mm_high.energy, nstate)
        energy_qm_high = traj_qm_high.energy

        grad_mm_all = np.repeat(traj_mm_all.grad, nstate, axis=0)
        grad_mm_high = np.repeat(traj_mm_high.grad, nstate, axis=0)
        grad_qm_high = traj_qm_high.grad

        traj.energy = energy_mm_all - energy_mm_high + energy_qm_high
        traj.grad = np.copy(grad_mm_all)
        traj.grad[:, highlevel, :] += -grad_mm_high + grad_qm_high

        traj.nac = np.zeros((len(traj_qm_high.nac), natom, 3))
        if len(traj_qm_high.nac) > 0:
            traj.nac[:, highlevel, :] = traj_qm_high.nac

        traj.soc = traj_qm_high.soc
        traj.err_energy = traj_qm_high.err_energy if hasattr(traj_qm_high, 'err_energy') else None
        traj.err_grad = traj_qm_high.err_grad if hasattr(traj_qm_high, 'err_grad') else None
        traj.err_nac = traj_qm_high.err_nac if hasattr(traj_qm_high, 'err_nac') else None
        traj.err_soc = traj_qm_high.err_soc if hasattr(traj_qm_high, 'err_soc') else None
        traj.status = np.amin([traj_qm_high.status, traj_mm_high.status, traj_mm_all.status])

        traj.qm_energy = traj_qm_high.energy
        traj.qm_grad = traj_qm_high.grad
        traj.qm_nac = traj_qm_high.nac
        traj.qm_soc = traj_qm_high.soc
        traj.energy_qm = energy_qm_high
        traj.energy_qm2_1 = 0
        traj.energy_qm2_2 = 0
        traj.energy_mm1 = traj_mm_high.energy[0]
        traj.energy_mm2 = traj_mm_all.energy[0]

        return traj

    def read_data(self, natom, ncharge=0):
        return self.mm_all.read_data(natom, ncharge=ncharge)
