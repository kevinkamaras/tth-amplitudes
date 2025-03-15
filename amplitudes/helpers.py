import numpy as np
import copy

epsLow  = np.array([[0, -1], [ 1, 0]])
epsHigh = np.array([[0,  1], [-1, 0]])

sig0 = np.array([[1,  0 ], [0 ,  1]])
sig1 = np.array([[0,  1 ], [1 ,  0]])
sig2 = np.array([[0, -1j], [1j,  0]])
sig3 = np.array([[1,  0 ], [0 , -1]])
pauli = [sig0, sig1, sig2, sig3]

class massless:
    '''spinors for a given massless momentum'''
    def __init__(self, p):
        p0 = p[0]
        p1 = p[1]
        p2 = p[2]
        p3 = p[3]
        m = np.emath.sqrt(np.vdot(p0,p0) - np.vdot(p1,p1) - np.vdot(p2,p2) - np.vdot(p3,p3))
        if np.real(m) >= 1:
            raise ValueError(f'momentum is not massless (m = {m})')
        pp = p0 + p3
        pm = p0 - p3
        pt = p1 + 1j*p2
        pb = p1 - 1j*p2
        theta = np.heaviside(-np.real(p0), 0)
        if pp == 0:
            raise ValueError('pp = 0')
        else:
            self.aket = ( 1j)**theta * np.array([[-pb / np.emath.sqrt(pp)], [np.emath.sqrt(pp)]])
            self.sbra = (-1j)**theta * np.array([[-pt / np.emath.sqrt(pp), np.emath.sqrt(pp)]])
            self.abra = np.matmul(epsHigh, self.aket).T
            self.sket = np.matmul(self.sbra, epsHigh.T).T
            self.vector = np.array([p0, p1, p2, p3])

    def momentum(self):
        return self.aket @ self.sbra
    
    def sketabra(self):
        return self.sket @ self.abra
        

class massive:
    '''spinors for a given massive momentum'''
    def __init__(self, p):
        p0 = p[0]
        p1 = p[1]
        p2 = p[2]
        p3 = p[3]
        E = p0
        P = np.sign(E) * np.emath.sqrt(p1**2 + p2**2 + p3**2)
        self.mass = np.emath.sqrt(E**2 - P**2)
        pp = P + p3
        pm = P - p3
        pt = p1 + 1j*p2
        pb = p1 - 1j*p2
        theta = np.heaviside(-np.real(E), 0)
        if pp == 0:
            raise ValueError('pp = 0')
        else:
            I1 =  np.emath.sqrt((E - P) / 2 / P) * np.array([[np.emath.sqrt(pp)], [ pt / np.emath.sqrt(pp)]])
            I2 =  np.emath.sqrt((E + P) / 2 / P) * np.array([[-pb / np.emath.sqrt(pp)], [np.emath.sqrt(pp)]])
            J1 =  np.emath.sqrt((E + P) / 2 / P) * np.array([[-pt / np.emath.sqrt(pp), np.emath.sqrt(pp)]])
            J2 = -np.emath.sqrt((E - P) / 2 / P) * np.array([[np.emath.sqrt(pp),  pb / np.emath.sqrt(pp)]])
            self.aket = ( 1j)**theta * np.hstack([I1, I2])
            self.sbra = (-1j)**theta * np.vstack([J1, J2])
            self.abra = np.matmul(epsHigh, self.aket  ).T
            self.sket = np.matmul(self.sbra, epsHigh.T).T
            self.aksb = p0 * sig0 - p1 * sig1 - p2 * sig2 - p3 * sig3
            self.skab = p0 * sig0 + p1 * sig1 + p2 * sig2 + p3 * sig3
            self.vector = np.array([p0, p1, p2, p3])
        
    def momentum(self):
        return self.aksb
    
    def sketabra(self):
        return self.skab

def minkowski(p):
    if len(p) != 4:
        raise TypeError('p is not a four vector')
    return p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2

def abraket(a, b):
    '''angle braket of two spinors. 0-, 1-, or 2-dimensional output'''
    braket = a.abra @ b.aket
    return braket

def sbraket(a, b):
    '''square braket of two spinors. 0-, 1-, or 2-dimensional output'''
    braket = a.sbra @ b.sket
    return braket

def PT4(g1, g2, g3, g4, negatives):
    '''Parke-Taylor amplitude for (--++) helicity combination'''
    gi, gj = list(negatives)
    amp = abraket(gi, gj)**4 / abraket(g1, g2) / abraket(g2, g3) / abraket(g3, g4) / abraket(g4, g1)
    return amp

def onShell(pi, pj, P, hcase, side):
    z = 0
    hati = copy.deepcopy(pi)
    hatj = copy.deepcopy(pj)
    match hcase[-1]:
        case '+':
            if side == 'left':
                z = - (pj.abra @ P[0].momentum() @ pj.sket) / (pi.abra @ P[0].momentum() @ pj.sket)
            elif len(P) == 2:
                z = (pi.abra @ P[0].momentum() @ pi.sket) / (pi.abra @ P[0].momentum() @ pj.sket)
            else:
                z = ((  pi.abra @ np.sum([p.momentum() for p in P[:-1]], axis=0) @ pi.sket
                    + P[-2].abra @ np.sum([p.momentum() for p in P[:-2]], axis=0) @ P[-2].sket)
                    / pi.abra @ np.sum([p.momentum() for p in P[:-1]], axis=0) @ pj.sket)
            z = z[0, 0]
            hati.sket = pi.sket - z * pj.sket
            hati.sbra = pi.sbra - z * pj.sbra
            hati.vector = np.array([(hati.abra @ sigma @ hati.sket)[0, 0] / 2 for sigma in pauli])
            hatj.abra = pj.abra + z * pi.abra
            hatj.aket = pj.aket + z * pi.aket
            hatj.vector = np.array([(hatj.abra @ sigma @ hatj.sket)[0, 0] / 2 for sigma in pauli])
        case '-':
            if side == 'left':
                z = (pj.abra @ P[0].momentum() @ pj.sket) / (pj.abra @ P[0].momentum() @ pi.sket)
            elif len(P) == 2:
                z = - (pi.abra @ P[0].momentum() @ pi.sket) / (pj.abra @ P[0].momentum() @ pi.sket)
            else:
                z = - ((  pi.abra @ np.sum([p.momentum() for p in P[:-1]], axis=0) @ pi.sket
                       + P[-2].abra @ np.sum([p.momentum() for p in P[:-2]], axis=0) @ P[-2].sket)
                       / pj.abra @ np.sum([p.momentum() for p in P[:-1]], axis=0) @ pi.sket)
            z = z[0, 0]
            hati.sket = pi.aket + z * pj.aket
            hati.sbra = pi.abra + z * pj.abra
            hati.vector = np.array([(hati.abra @ sigma @ hati.sket)[0, 0] / 2 for sigma in pauli])
            hatj.abra = pj.sbra - z * pi.sbra
            hatj.aket = pj.sket - z * pi.sket
            hatj.vector = np.array([(hatj.abra @ sigma @ hatj.sket)[0, 0] / 2 for sigma in pauli])
    return hati, hatj