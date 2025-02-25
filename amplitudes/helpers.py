import numpy as np

mtop = 1
epsLow  = np.array([[0, -1], [ 1, 0]])
epsHigh = np.array([[0,  1], [-1, 0]])

sig0 = np.array([[1,  0 ], [0 ,  1]])
sig1 = np.array([[0,  1 ], [1 ,  0]])
sig2 = np.array([[0, -1j], [1j,  0]])
sig3 = np.array([[1,  0 ], [0 , -1]])
pauli = [sig0, sig1, sig2, sig3]

class massless:
    '''spinors for a given massless momentum'''
    def __init__(self, p0, p1, p2, p3):
        m = np.emath.sqrt(np.vdot(p0,p0) - np.vdot(p1,p1) - np.vdot(p2,p2) - np.vdot(p3,p3))
        if m >= 1e-5:
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
            self.momentum = p0 * sig0 - p1 * sig1 - p2 * sig2 - p3 * sig3
            self.sketabra = p0 * sig0 + p1 * sig1 + p2 * sig2 + p3 * sig3
            self.vector = np.array([p0, p1, p2, p3])

class massive:
    '''spinors for a given massive momentum'''
    def __init__(self, p0, p1, p2, p3):
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
            self.momentum = p0 * sig0 - p1 * sig1 - p2 * sig2 - p3 * sig3
            self.sketabra = p0 * sig0 + p1 * sig1 + p2 * sig2 + p3 * sig3
            self.vector = np.array([p0, p1, p2, p3])

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

# def PT3(g1, g2, g3, invert=False):
#     '''Parke-Taylor amplitude for (--+) or (++-) helicity combinations'''
#     if invert:
#         amp = sbraket(g1, g2)**3 / sbraket(g2, g3) / sbraket(g3, g1)
#     else:
#         amp = abraket(g1, g2)**3 / abraket(g2, g3) / abraket(g3, g1)
#     return amp

def PT4(g1, g2, g3, g4, negatives):
    '''Parke-Taylor amplitude for (--++) helicity combination'''
    gi, gj = list(negatives)
    amp = abraket(gi, gj)**4 / abraket(g1, g2) / abraket(g2, g3) / abraket(g3, g4) / abraket(g4, g1)
    return amp

def onShell(pi, pj, P, hcase, side):
    z = 0
    match hcase[-2]:
        case '-':
            if side == 'left':
                z = (pj.abra @ P[0].momentum @ pj.sket) / (pj.abra @ P[0].momentum @ pi.sket)
            elif len(P) == 2:
                z = - (pi.abra @ P[0].momentum @ pi.sket) / (pj.abra @ P[0].momentum @ pi.sket)
            else:
                z = - ((  pi.abra @ np.sum([p.momentum for p in P[:-1]], axis=0) @ pi.sket
                       + P[-2].abra @ np.sum([p.momentum for p in P[:-2]], axis=0) @ P[-2].sket)
                       / pj.abra @ np.sum([p.momentum for p in P[:-1]], axis=0) @ pi.sket)
            z = z[0, 0]
            ivector = np.array([((pi.abra + z * pj.abra) @ sigma @ pj.sket)[0, 0] / 2 for sigma in pauli])
            jvector = np.array([(pi.abra @ sigma @ (pj.sket - z * pi.sket))[0, 0] / 2 for sigma in pauli])
        case '+':
            if side == 'left':
                z = - (pj.abra @ P[0].momentum @ pj.sket) / (pi.abra @ P[0].momentum @ pj.sket)
            elif len(P) == 2:
                z = (pi.abra @ P[0].momentum @ pi.sket) / (pi.abra @ P[0].momentum @ pj.sket)
            else:
                z = ((  pi.abra @ np.sum([p.momentum for p in P[:-1]], axis=0) @ pi.sket
                    + P[-2].abra @ np.sum([p.momentum for p in P[:-2]], axis=0) @ P[-2].sket)
                    / pi.abra @ np.sum([p.momentum for p in P[:-1]], axis=0) @ pj.sket)
            z = z[0, 0]
            ivector = np.array([(pi.abra @ sigma @ (pi.sket - z * pj.sket))[0, 0] / 2 for sigma in pauli])
            jvector = np.array([((pj.abra + z * pi.abra) @ sigma @ pj.sket)[0, 0] / 2 for sigma in pauli])
    hati = massless(*ivector)
    hatj = massless(*jvector)
    return hati, hatj

def ggg(g1, g2, g3, hcase):
    '''from Campbell2023'''
    match hcase:
        case ['-', '-', '+']:
            return abraket(g1, g2)**3 / abraket(g2, g3) / abraket(g3, g1)
        case ['+', '+', '-']:
            return sbraket(g1, g2)**3 / sbraket(g2, g3) / sbraket(g3, g1)
        case ['-', '+', '-']:
            return abraket(g3, g1)**3 / abraket(g1, g2) / abraket(g2, g3)
        case ['+', '-', '+']:
            return sbraket(g3, g1)**3 / sbraket(g1, g2) / sbraket(g2, g3)
        case ['+', '-', '-']:
            return abraket(g2, g3)**3 / abraket(g1, g2) / abraket(g3, g1)
        case ['-', '+', '+']:
            return sbraket(g2, g3)**3 / sbraket(g1, g2) / sbraket(g3, g1)
    return 0

# def ggg(g1, g2, g3, hcase):
#     '''from Campbell2023'''
#     match hcase:
#         case ['-', '-', '+']:
#             return PT3(g1, g2, g3)
#         case ['+', '+', '-']:
#             return PT3(g1, g2, g3, invert=True)
#         case ['-', '+', '-']:
#             return PT3(g3, g1, g2)
#         case ['+', '-', '+']:
#             return PT3(g3, g1, g2, invert=True)
#         case ['+', '-', '-']:
#             return PT3(g2, g3, g1)
#         case ['-', '+', '+']:
#             return PT3(g2, g3, g1, invert=True)
#     return 0

def gqq(g1, q2, qbar3, hcase):
    '''from Arkani-Hamed2021'''
    match hcase:
        case ['-', '-', '+']:
            if abs(abraket(g1, q2)) <= 1e-6:
                return 0
            return abraket(g1, q2)**2 / abraket(q2, qbar3)
        case ['+', '+', '-']:
            if abs(sbraket(g1, q2)) <= 1e-6:
                return 0
            return sbraket(g1, q2)**2 / sbraket(q2, qbar3)
        case ['-', '+', '-']:
            if abs(abraket(qbar3, g1)) <= 1e-6:
                return 0
            return abraket(qbar3, g1)**2 / abraket(q2, qbar3)
        case ['+', '-', '+']:
            if abs(sbraket(qbar3, g1)) <= 1e-6:
                return 0
            return sbraket(qbar3, g1)**2 / sbraket(q2, qbar3)
    return 0

def ttg(t1, tbar2, g3, hcase, ref):
    match hcase:
        case '+':
            amp = (ref.abra @ t1.momentum @ g3.sket) * sbraket(t1, tbar2) / (t1.mass * abraket(ref, g3))
            return amp
        case '-':
            amp = (g3.abra @ t1.momentum @ ref.sket) * sbraket(t1, tbar2) / (t1.mass * sbraket(g3, ref))
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
    amp = None
    return amp

def ttgg(t1, tbar2, g3, g4, hcase):
    '''from Campbell2023'''
    p34 = g3.vector + g4.vector
    s34 = minkowski(p34)
    match hcase:
        case ['+', '+']:
            amp = t1.mass * sbraket(g3, g4) * abraket(t1, tbar2) / abraket(g3, g4) / (g3.abra @ t1.momentum @ g3.sket)
            return amp
        case ['-', '-']:
            amp = t1.mass * abraket(g3, g4) * sbraket(t1, tbar2) / sbraket(g3, g4) / (g3.abra @ t1.momentum @ g3.sket)
            return amp
        case ['+', '-']:
            amp = ((g4.abra @ t1.momentum @ g3.sket)
                   * (sbraket(t1, g3) @ abraket(g4, tbar2) + abraket(t1, g4) @ sbraket(g3, tbar2))
                   / (t1.mass * (g3.abra @ t1.momentum @ g3.sket) * s34))
            return amp
        case ['-', '+']:
            amp = ((g3.abra @ t1.momentum @ g4.sket)
                   * (abraket(t1, g3) @ sbraket(g4, tbar2) + sbraket(t1, g4) @ abraket(g3, tbar2))
                   / (t1.mass * (g3.abra @ t1.momentum @ g3.sket) * s34))
            return amp
    raise ValueError('missing gluon helicities')

def ttqq(t1, tbar2, q3, qbar4, hcase):
    '''from Campbell2023'''
    p34 = q3.vector + qbar4.vector
    s34 = np.vdot(p34, p34)
    match hcase:
        case ['+', '-']:
            amp = (abraket(t1, q3) @ sbraket(qbar4, tbar2) + sbraket(t1, qbar4) @ abraket(q3, tbar2)) / s34
            return amp
        case ['-', '+']:
            amp = (abraket(t1, qbar4) @ sbraket(q3, tbar2) + sbraket(t1, q3) @ abraket(qbar4, tbar2)) / s34
            return amp
    return [[0, 0], [0, 0]]