import numpy as np

mtop = 1
eps = np.array([[0, 1], [-1, 0]])

class massless:
    '''spinors for a given massless momentum'''
    def __init__(self, p0, p1, p2, p3):
        m2 = abs(p0)**2 - abs(p1)**2 - abs(p2)**2 - abs(p3)**2
        if m2 >= 1e-10:
            raise ValueError('momentum is not massless')
        pp = p0 + p3
        pm = p0 - p3
        pt = p1 + 1j*p2
        pb = p1 - 1j*p2
        theta = (1 + np.sign(-p0)) / 2
        if pp == 0:
            raise ValueError('pp = 0')
        else:
            self.aket = ( 1j)**theta * np.array([[-pb / np.sqrt(pp), np.sqrt(pp)]]).T
            self.sbra = (-1j)**theta * np.array([[-pt / np.sqrt(pp), np.sqrt(pp)]])
            self.abra = np.matmul(eps, self.aket).T
            self.sket = np.matmul(self.sbra, eps).T

class massive:
    '''spinors for a given massive momentum'''
    def __init__(self, p0, p1, p2, p3):
        E = p0
        P = np.sign(E) * np.sqrt(abs(p1)**2 + abs(p2)**2 + abs(p3)**2)
        pp = p0 + p3
        pm = p0 - p3
        pt = p1 + 1j*p2
        pb = p1 - 1j*p2
        if pp == 0:
            raise ValueError('pp = 0')
        else:
            I1 = np.sqrt((E - P) / 2 / P) * np.array([[np.sqrt(pp),  pt / np.sqrt(pp)]]).T
            I2 = np.sqrt((E + P) / 2 / P) * np.array([[-pb / np.sqrt(pp), np.sqrt(pp)]]).T
            J1 = np.sqrt((E + P) / 2 / P) * np.array([[-pt / np.sqrt(pp), np.sqrt(pp)]])
            J2 = np.sqrt((E - P) / 2 / P) * np.array([[np.sqrt(pp),  pb / np.sqrt(pp)]])
            self.aket = np.hstack([I1, I2])
            self.sbra = np.vstack([J1, J2])
            self.abra = np.matmul(eps, self.aket).T  
            self.sket = np.matmul(self.sbra, eps).T

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

def onShell(pi, pj, P, hcase):
    match hcase[-2]:
        case ['+']:
            z = 0
            hati = pi + z * pj
            hatj = pj - z * pi
        case ['+']:
            z = 0
            hati = pi - z * pj
            hatj = pj + z * pi
    return hati, hatj

def ggg(g1, g2, g3, hcase):
    '''from Campbell2023'''
    g1 = massless(*g1)
    g2 = massless(*g2)
    g3 = massless(*g3)
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
    t1 = massive(*t1)
    tbar2 = massive(*tbar2)
    g3 = massless(*g3)
    ref = massless(*ref)
    match hcase:
        case ['+']:
            amp = (abraket(ref, t1) @ sbraket(t1, g3)) * sbraket(t1, tbar2) / (mtop * abraket(ref, g3))
            return amp
        case ['-']:
            amp = (abraket(g3, t1) @ sbraket(t1, ref)) * sbraket(t1, tbar2) / (mtop * abraket(g3, ref))
            return amp
    raise ValueError('missing gluon helicity')

def tth(t1, tbar2, h3):
    t1 = massive(t1)
    tbar2 = massive(tbar2)
    amp = abraket(t1, tbar2) + sbraket(t1, tbar2)
    return amp

def gggg(g1, g2, g3, g4, hcase):
    g1 = massless(*g1)
    g2 = massless(*g2)
    g3 = massless(*g3)
    g4 = massless(*g4)
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
    t1 = massive(*t1)
    tbar2 = massive(*tbar2)
    g3 = massless(*g3)
    g4 = massless(*g4)
    match hcase:
        case ['+', '+']:
            amp = mtop * sbraket(g3, g4) * abraket(t1, tbar2) / abraket(g3, g4) / (abraket(g3, tbar2) @ sbraket(tbar2, g3))
            return amp
        case ['-', '-']:
            amp = mtop * abraket(g3, g4) * sbraket(t1, tbar2) / sbraket(g3, g4) / (abraket(g3, tbar2) @ sbraket(tbar2, g3))
            return amp
        case ['+', '-']:
            amp = - {(abraket(g4, t1) @ sbraket(t1, g3)) * (abraket(g4, tbar2) @ sbraket(tbar2, g3)) * abraket(t1, tbar2)
                     / mtop * (abraket(g3, tbar2) @ sbraket(tbar2, g3)) * s34}
            return amp
        case ['-', '+']:
            amp = - {(sbraket(g4, t1) @ abraket(t1, g3)) * (sbraket(g4, tbar2) @ abraket(tbar2, g3)) * sbraket(t1, tbar2)
                     / mtop * (sbraket(g3, tbar2) @ abraket(tbar2, g3)) * s34}
            return amp
    raise ValueError('missing gluon helicities')

def ttqq(t1, tbar2, q3, qbar4, hcase):
    '''from Campbell2023'''
    p34 = q3 + qbar4
    s34 = np.vdot(p34, p34)
    t1 = massive(*t1)
    tbar2 = massive(*tbar2)
    q3 = massless(*q3)
    qbar4 = massless(*qbar4)
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