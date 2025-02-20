import numpy as np

mtop = 1
epsLow  = np.array([[0, -1], [ 1, 0]])
epsHigh = np.array([[0,  1], [-1, 0]])

sig0 = np.array([[1,  0 ], [0 ,  1]])
sig1 = np.array([[0,  1 ], [1 ,  0]])
sig2 = np.array([[0, -1j], [1j,  0]])
sig3 = np.array([[1,  0 ], [0 , -1]])
pauli = [sig0, -sig1, -sig2, -sig3]

class massless:
    '''spinors for a given massless momentum'''
    def __init__(self, p0, p1, p2, p3):
        m2 = np.vdot(p0,p0) - np.vdot(p1,p1) - np.vdot(p2,p2) - np.vdot(p3,p3)
        if m2 >= 1e-10:
            raise ValueError('momentum is not massless')
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
            self.momentum = p0 * sig0 - p1 * sig1 - p2 * sig2 - p3 * sig3
            self.sketabra = p0 * sig0 + p1 * sig1 + p2 * sig2 + p3 * sig3
            self.vector = np.array([p0, p1, p2, p3])

class massive:
    '''spinors for a given massive momentum'''
    def __init__(self, p0, p1, p2, p3):
        E = p0
        P = np.sign(E) * np.emath.sqrt(np.vdot(p1,p1) + np.vdot(p2,p2) + np.vdot(p3,p3))
        self.mass = np.emath.sqrt(E**2 - P**2)
        pp = P + p3
        pm = p0 - p3
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
            self.momentum = p0 * sig0 - p1 * sig1 - p2 * sig2 - p3 * sig3
            self.sketabra = p0 * sig0 + p1 * sig1 + p2 * sig2 + p3 * sig3
            self.vector = np.array([p0, p1, p2, p3])

def abraket(a, b):
    '''angle braket of two spinors. 0-, 1-, or 2-dimensional output'''
    braket = a.abra @ b.aket
    return braket

def sbraket(a, b):
    '''square braket of two spinors. 0-, 1-, or 2-dimensional output'''
    braket = a.sbra @ b.sket
    return braket

def PT3(g1, g2, g3, invert=False):
    '''Parke-Taylor amplitude for (--+) or (++-) helicity combinations'''
    if invert:
        amp = sbraket(g1, g2)**3 / sbraket(g2, g3) / sbraket(g3, g1)
    else:
        amp = abraket(g1, g2)**3 / abraket(g2, g3) / abraket(g3, g1)
    return amp

def PT4(g1, g2, g3, g4, negatives):
    '''Parke-Taylor amplitude for (--++) helicity combination'''
    gi, gj = list(negatives)
    amp = abraket(gi, gj)**4 / abraket(g1, g2) / abraket(g2, g3) / abraket(g3, g4) / abraket(g4, g1)
    return amp

def onShell(pi, pj, P, hcase, side):
    z = 0
    bispinori = np.empty((2,2))
    bispinorj = np.empty((2,2))
    match hcase[-2]:
        case ['+']:
            if side == 'left':
                z = (P[1].abra @ P[0].momentum @ P[1].sket) / (pj.abra @ P[0].momentum @ pi.sket)
            elif len(P) == 2:
                z = - (P[1].abra @ P[0].momentum @ P[1].sket) / (pj.abra @ P[0].momentum @ pi.sket)
            else:
                z = - (  P[-1].abra @ np.sum([p.momentum for p in P[:-1]]) @ P[-1].sket
                       + P[-2].abra @ np.sum([p.momentum for p in P[:-2]]) @ P[-2].sket) / pj.sbra @ np.sum([p.momentum for p in P[:-1]]) @ P[-1].sket
            bispinori = (pi.aket + z * pj.aket) @ pi.sbra 
            bispinorj = pj.aket @ (pj.sbra - z * pi.sbra)
        case ['-']:
            if side == 'left':
                z = - (P[1].abra @ P[0].momentum @ P[1].sket) / (pi.abra @ P[0].momentum @ pj.sket)
            elif len(P) == 2:
                z = (P[1].abra @ P[0].momentum @ P[1].sket) / (pi.abra @ P[0].momentum @ pj.sket)
            else:
                z = (  P[-1].abra @ np.sum([p.momentum for p in P[:-1]]) @ P[-1].sket
                    + P[-2].abra @ np.sum([p.momentum for p in P[:-2]]) @ P[-2].sket) / pi.sbra @ np.sum([p.momentum for p in P[:-1]]) @ pj.sket
            bispinori = pi.aket @ (pi.sbra - z * pj.sbra)
            bispinorj = (pj.aket + z * pi.aket) @ pj.sbra 
    hati = massless(*np.array([np.trace(bispinori @ sigma) /2 for sigma in pauli]))
    hatj = massless(*np.array([np.trace(bispinorj @ sigma) /2 for sigma in pauli]))
    return hati, hatj

def ggg(g1, g2, g3, hcase):
    '''from Campbell2023'''
    match hcase:
        case ['-', '-', '+']:
            return PT3(g1, g2, g3)
        case ['+', '+', '-']:
            return PT3(g1, g2, g3, invert=True)
        case ['-', '+', '-']:
            return PT3(g3, g1, g2)
        case ['+', '-', '+']:
            return PT3(g3, g1, g2, invert=True)
        case ['+', '-', '-']:
            return PT3(g2, g3, g1)
        case ['-', '+', '+']:
            return PT3(g2, g3, g1, invert=True)
    return 0

def qqg(g1, q2, qbar3, hcase):
    amp = ggg(g1, q2, qbar3, hcase)
    return amp

def ttg(t1, tbar2, g3, hcase, ref):
    match hcase:
        case ['+']:
            amp = (abraket(ref, t1) @ sbraket(t1, g3)) * sbraket(t1, tbar2) / (mtop * abraket(ref, g3))
            return amp
        case ['-']:
            amp = (abraket(g3, t1) @ sbraket(t1, ref)) * sbraket(t1, tbar2) / (mtop * abraket(g3, ref))
            return amp
    raise ValueError('missing gluon helicity')

def tth(t1, tbar2, h3):
    amp = abraket(t1, tbar2) + sbraket(t1, tbar2)
    return amp

def gggg(g1, g2, g3, g4, hcase):
    momenta = [g1, g2, g3, g4]
    negatives = momenta[hcase == '-']
    # if not MHV, amplitude is zero
    if len(negatives) != 2:
        return 0
    else:
        return PT4(g1, g2, g3, g4, negatives)

def qqgg(g1, q2, qbar3, g4, hcase):
    amp = gggg(g1, q2, qbar3, g4, hcase)
    return amp

def ttgg(t1, tbar2, g3, g4, hcase):
    p34 = g3 + g4
    s34 = np.vdot(p34, p34)
    match hcase:
        case ['+', '+']:
            amp = mtop * sbraket(g3, g4) * abraket(t1, tbar2) / abraket(g3, g4) / (abraket(g3, tbar2) @ sbraket(tbar2, g3))
            return amp
        case ['-', '-']:
            amp = mtop * abraket(g3, g4) * sbraket(t1, tbar2) / sbraket(g3, g4) / (abraket(g3, tbar2) @ sbraket(tbar2, g3))
            return amp
        case ['+', '-']:
            amp = - ((abraket(g4, t1) @ sbraket(t1, g3)) * (abraket(g4, tbar2) @ sbraket(tbar2, g3)) * abraket(t1, tbar2)
                     / mtop * (abraket(g3, tbar2) @ sbraket(tbar2, g3)) * s34)
            return amp
        case ['-', '+']:
            amp = - ((sbraket(g4, t1) @ abraket(t1, g3)) * (sbraket(g4, tbar2) @ abraket(tbar2, g3)) * sbraket(t1, tbar2)
                     / mtop * (sbraket(g3, tbar2) @ abraket(tbar2, g3)) * s34)
            return amp
    raise ValueError('missing gluon helicities')

def ttqq(t1, tbar2, q3, qbar4, hcase):
    '''from Campbell2023'''
    p34 = q3 + qbar4
    s34 = np.vdot(p34, p34)
    match hcase:
        case ['+', '-']:
            amp = (abraket(t1, q3) @ sbraket(qbar4, tbar2) + sbraket(t1, qbar4) @ abraket(q3, tbar2)) / s34
            return amp
        case ['-', '+']:
            amp = (abraket(t1, qbar4) @ sbraket(q3, tbar2) + sbraket(t1, q3) @ abraket(qbar4, tbar2)) / s34
            return amp
    return 0

def ttggg(t1, tbar2, g3, g4, g5, hcase):
    amp = None
    return amp

def ttqqg(t1, tbar2, q3, qbar4, g5, hcase):
    amp = None
    return amp