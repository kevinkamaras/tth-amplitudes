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
    braket = np.matmul(a.abra, b.aket)
    return braket

def sbraket(a, b):
    '''square braket of two spinors. 0-, 1-, or 2-dimensional output'''
    braket = np.matmul(a.sbra, b.sket)
    return braket

def BCFW(i, j, hcase):
    shift = None
    return shift

def onShell(shift, p):
    hati = None
    hatj = None
    return hati, hatj

def ggg(g1, g2, g3, hcase):
    amp = None
    return amp

def qqg(g1, q2, qbar3, hcase):
    amp = None
    return amp

def ttg(t1, tbar2, g3, hcase):
    amp = None
    return amp

def tth(t1, tbar2, h3):
    amp = None
    return amp

def gggg(g1, g2, g3, g4, hcase):
    amp = None
    return amp

def qqgg(g1, q2, qbar3, g4, hcase):
    amp = None
    return amp

def ttgg(t1, tbar2, g3, g4, hcase):
    amp = None
    return amp

def ttggg(t1, tbar2, g3, g4, g5, hcase):
    amp = None
    return amp