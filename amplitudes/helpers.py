import numpy as np
import copy
import scipy.optimize as opt

mTop = 171
mHiggs = 125

epsLow  = np.array([[0, -1], [ 1, 0]])
epsHigh = np.array([[0,  1], [-1, 0]])

sig0 = np.array([[1,  0 ], [0 ,  1]])
sig1 = np.array([[0,  1 ], [1 ,  0]])
sig2 = np.array([[0, -1j], [1j,  0]])
sig3 = np.array([[1,  0 ], [0 , -1]])
pauli = [sig0, sig1, sig2, sig3]
pauliBar = [sig0, -sig1, -sig2, -sig3]

class massless:
    '''spinors for a given massless momentum'''
    def __init__(self, p):
        self.valid = True
        p0 = p[0]
        p1 = p[1]
        p2 = p[2]
        p3 = p[3]
        m = np.emath.sqrt(p0**2 - p1**2 - p2**2 - p3**2)
        if abs(m) >= 1e-3:
            print(f'momentum is not massless (m = {abs(m)}), disregarding amplitude')
            self.valid = False
        pp = p0 + p3
        pm = p0 - p3
        pt = p1 + 1j*p2
        pb = p1 - 1j*p2
        theta = np.heaviside(-np.real(p0), 0)
        if abs(pp) >= 1e-10:
            self.aket = ( 1j)**theta * np.array([[-pb / np.emath.sqrt(pp)], [np.emath.sqrt(pp)]])
            self.sbra = (-1j)**theta * np.array([[-pt / np.emath.sqrt(pp), np.emath.sqrt(pp)]])
        elif abs(pm) >= 1e-10:
            self.aket = ( 1j)**theta * np.array([[- np.emath.sqrt(pm)], [pt / np.emath.sqrt(pm)]])
            self.sbra = (-1j)**theta * np.array([[- np.emath.sqrt(pm), pb / np.emath.sqrt(pm)]])
        else:
            self.aket = 1 / np.emath.sqrt(2 * p1) * np.array([[- pb], [pt]])
            self.sbra = 1 / np.emath.sqrt(2 * p1) * np.array([[- pt, pb]])
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
        self.valid = True
        p0 = p[0]
        p1 = p[1]
        p2 = p[2]
        p3 = p[3]
        E = p0
        if np.real(E) > -1e-5 or (abs(np.real(E)) <= 1e-5 and np.imag(E) >= -1e-5):
            signE = 1
        elif np.real(E) < -1e-5 or (abs(np.real(E)) <= 1e5 and np.imag(E) < -1e-5):
            signE = -1
        P = signE * np.emath.sqrt(p1**2 + p2**2 + p3**2)
        self.mass = np.emath.sqrt(E**2 - P**2)
        pp = P + p3
        pm = P - p3
        pt = p1 + 1j*p2
        pb = p1 - 1j*p2
        theta = np.heaviside(-np.real(E), 0)
        thetap0 = np.heaviside(-np.real(p0), 0)
        if np.array([abs(p1) <= 1e-10, abs(p2) <= 1e-10, abs(p3) <= 1e-10]).all():
            param = 6
        elif abs(P) <= 1e-5:
            if abs(p3) >= 1e-5:
                param = 3
            elif abs(pt) <= 1e-5:
                param = 4
            else:
                param = 5
        elif abs(pp) <= 1e-5:
            param = 2
        else:
            param = 1   
        match param:
            case 1:
                I1 =  np.emath.sqrt((E - P) / 2 / P) * np.array([[np.emath.sqrt(pp)], [pt / np.emath.sqrt(pp)]])
                I2 =  np.emath.sqrt((E + P) / 2 / P) * np.array([[-pb / np.emath.sqrt(pp)], [np.emath.sqrt(pp)]])
                J1 =  np.emath.sqrt((E + P) / 2 / P) * np.array([[-pt / np.emath.sqrt(pp), np.emath.sqrt(pp)]])
                J2 = -np.emath.sqrt((E - P) / 2 / P) * np.array([[np.emath.sqrt(pp),  pb / np.emath.sqrt(pp)]])
                self.aket = ( 1j)**theta * np.hstack([I1, I2])
                self.sbra = (-1j)**theta * np.vstack([J1, J2])
            case 2:
                I1 =  np.emath.sqrt((E - P) / 2 / P) * np.array([[pb / np.emath.sqrt(pm)], [np.emath.sqrt(pm)]])
                I2 =  np.emath.sqrt((E + P) / 2 / P) * np.array([[-np.emath.sqrt(pm)], [pt / np.emath.sqrt(pm)]])
                J1 =  np.emath.sqrt((E + P) / 2 / P) * np.array([[-np.emath.sqrt(pm), pt / np.emath.sqrt(pm)]])
                J2 = -np.emath.sqrt((E - P) / 2 / P) * np.array([[pt / np.emath.sqrt(pm),  np.emath.sqrt(pm)]])
                self.aket = ( 1j)**theta * np.hstack([I1, I2])
                self.sbra = (-1j)**theta * np.vstack([J1, J2])
            case 3:
                I1 =  p0 / np.emath.sqrt(p0 - pp) * np.array([[0], [-np.emath.sqrt(pt / pb)]])
                I2 =  np.emath.sqrt(pb) * np.array([[np.emath.sqrt((p0 - pp) / pt)], [-np.emath.sqrt(pt / (p0 - pp))]])
                J1 =  np.emath.sqrt(pt) * np.array([[np.emath.sqrt((p0 - pp) / pb), -np.emath.sqrt(pb / (p0 - pp))]])
                J2 = -p0 / np.emath.sqrt(p0 - pp) * np.array([[0, -np.emath.sqrt(pb / pt)]])
                self.aket = ( 1j)**thetap0 * np.hstack([I1, I2])
                self.sbra = (-1j)**thetap0 * np.vstack([J1, J2])
            case 4:
                I1 =  1 / np.emath.sqrt(2 * p1) * np.array([[-p0], [p0]])
                I2 =  1 / np.emath.sqrt(2 * p1) * np.array([[-pb + p0], [pt - p0]])
                J1 =  1 / np.emath.sqrt(2 * p1) * np.array([[-pt - p0, pb - p0]])
                J2 = -1 / np.emath.sqrt(2 * p1) * np.array([[2*pt - p0, 2 * pb - p0]])
                self.aket = ( 1j)**thetap0 * np.hstack([I1, I2])
                self.sbra = (-1j)**thetap0 * np.vstack([J1, J2])
            case 5:
                I1 =  1 / np.emath.sqrt(2 * p1) * np.array([[p0], [-p0]])
                I2 =  1 / np.emath.sqrt(2 * p1) * np.array([[-pb + p0], [pt - p0]])
                J1 =  1 / np.emath.sqrt(2 * p1) * np.array([[-pt + p0, pb +  p0]])
                J2 = -1 / np.emath.sqrt(2 * p1) * np.array([[2*pt - p0, 2 * pb - p0]])
                self.aket = ( 1j)**thetap0 * np.hstack([I1, I2])
                self.sbra = (-1j)**thetap0 * np.vstack([J1, J2])
            case 6:
                I1 =  np.array([[np.emath.sqrt(p0)], [0]])
                I2 =  np.array([[0], [np.emath.sqrt(p0)]])
                J1 =  np.array([[0, np.emath.sqrt(p0)]])
                J2 = -np.array([[np.emath.sqrt(p0), 0]])
                self.aket = ( 1j)**thetap0 * np.hstack([I1, I2])
                self.sbra = (-1j)**thetap0 * np.vstack([J1, J2])
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

def dot(p, q):
    if len(p) != 4 or len(q) != 4:
        raise TypeError('please choose two four-vectors')
    return p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]

def boost(momenta, v, phi, theta):
    bx = np.sin(theta) * np.cos(phi)
    by = np.sin(theta) * np.sin(phi)
    bz = np.cos(theta)

    beta = v * np.array([bx, by, bz])
    gamma = 1 / np.sqrt(1 - np.dot(beta, beta))
    bx = beta[0]
    by = beta[1]
    bz = beta[2]
    G = gamma**2 / (1 + gamma)

    L = np.array([[       gamma,  - gamma * bx,  - gamma * by,  - gamma * bz],
                    [- gamma * bx, 1 + G * bx**2,   G * bx * by,   G * bx * bz],
                    [- gamma * by,   G * bx * by, 1 + G * by**2,   G * by * bz],
                    [- gamma * bz,   G * bx * bz,   G * by * bz, 1 + G * bz**2]])

    momenta_b = [L @ p for p in momenta]
    return momenta_b

def tthggMomenta(angles):
    phi1   = angles[0]
    theta1 = angles[1]
    phi2   = angles[2]
    theta2 = angles[3]
    phi4   = angles[4]
    theta4 = angles[5]
    phi5   = angles[6]
    theta5 = angles[7]

    def F(x):
        p1 = x[0]
        p2 = x[1]
        p4 = x[2]
        p5 = x[3]
        F = np.empty(4)
        F[0] = np.emath.sqrt(p1**2 + mTop**2) + np.emath.sqrt(p2**2 + mTop**2) + p4 + p5 + mHiggs
        F[1] = (p1 * np.sin(theta1) * np.cos(phi1)
                + p2 * np.sin(theta2) * np.cos(phi2)
                + p4 * np.sin(theta4) * np.cos(phi4)
                + p5 * np.sin(theta5) * np.cos(phi5))
        F[2] = (p1 * np.sin(theta1) * np.sin(phi1)
                + p2 * np.sin(theta2) * np.sin(phi2)
                + p4 * np.sin(theta4) * np.sin(phi4)
                + p5 * np.sin(theta5) * np.sin(phi5))
        F[3] = (p1 * np.cos(theta1)
                + p2 * np.cos(theta2)
                + p4 * np.cos(theta4)
                + p5 * np.cos(theta5))
        return F

    guess = np.array([100, -100, 100, -100])
    p_opt = opt.newton_krylov(F, guess)

    p1 = p_opt[0]
    p2 = p_opt[1]
    p4 = p_opt[2]
    p5 = p_opt[3]

    p1x = p1 * np.sin(theta1) * np.cos(phi1)
    p1y = p1 * np.sin(theta1) * np.sin(phi1)
    p1z = p1 * np.cos(theta1)

    p2x = p2 * np.sin(theta2) * np.cos(phi2)
    p2y = p2 * np.sin(theta2) * np.sin(phi2)
    p2z = p2 * np.cos(theta2)

    p4x = p4 * np.sin(theta4) * np.cos(phi4)
    p4y = p4 * np.sin(theta4) * np.sin(phi4)
    p4z = p4 * np.cos(theta4)

    p5x = p5 * np.sin(theta5) * np.cos(phi5)
    p5y = p5 * np.sin(theta5) * np.sin(phi5)
    p5z = p5 * np.cos(theta5)

    E1 = np.sqrt(p1**2 + mTop**2)
    E2 = np.sqrt(p2**2 + mTop**2)
    E4 = p4
    E5 = p5

    t1    = np.array([E1, p1x, p1y, p1z])
    tbar2 = np.array([E2, p2x, p2y, p2z])
    h3    = np.array([mHiggs, 0, 0, 0])
    g4    = np.array([E4, p4x, p4y, p4z])
    g5    = np.array([E5, p5x, p5y, p5z])

    momenta = [t1, tbar2, h3, g4, g5]
    return momenta

def tthgggMomenta(angles, p1):
    phi1   = angles[0]
    theta1 = angles[1]
    phi2   = angles[2]
    theta2 = angles[3]
    phi4   = angles[4]
    theta4 = angles[5]
    phi5   = angles[6]
    theta5 = angles[7]
    phi6   = angles[8]
    theta6 = angles[9]

    def F(x):
        p2 = x[0]
        p4 = x[1]
        p5 = x[2]
        p6 = x[3]
        F = np.empty(4)
        F[0] = (np.emath.sqrt(p1**2 + mTop**2) + np.emath.sqrt(p2**2 + mTop**2)
                + p4 + p5 + p6 + mHiggs)
        F[1] = (p1 * np.sin(theta1) * np.cos(phi1)
                + p2 * np.sin(theta2) * np.cos(phi2)
                + p4 * np.sin(theta4) * np.cos(phi4)
                + p5 * np.sin(theta5) * np.cos(phi5)
                + p6 * np.sin(theta6) * np.cos(phi6))
        F[2] = (p1 * np.sin(theta1) * np.sin(phi1)
                + p2 * np.sin(theta2) * np.sin(phi2)
                + p4 * np.sin(theta4) * np.sin(phi4)
                + p5 * np.sin(theta5) * np.sin(phi5)
                + p6 * np.sin(theta6) * np.sin(phi6))
        F[3] = (p1 * np.cos(theta1)
                + p2 * np.cos(theta2)
                + p4 * np.cos(theta4)
                + p5 * np.cos(theta5)
                + p6 * np.cos(theta6))
        return F
    
    guess = np.array([-p1, -p1, -p1, -p1])
    p_opt = opt.newton_krylov(F, guess)

    p2 = p_opt[0]
    p4 = p_opt[1]
    p5 = p_opt[2]
    p6 = p_opt[3]

    p1x = p1 * np.sin(theta1) * np.cos(phi1)
    p1y = p1 * np.sin(theta1) * np.sin(phi1)
    p1z = p1 * np.cos(theta1)

    p2x = p2 * np.sin(theta2) * np.cos(phi2)
    p2y = p2 * np.sin(theta2) * np.sin(phi2)
    p2z = p2 * np.cos(theta2)

    p4x = p4 * np.sin(theta4) * np.cos(phi4)
    p4y = p4 * np.sin(theta4) * np.sin(phi4)
    p4z = p4 * np.cos(theta4)

    p5x = p5 * np.sin(theta5) * np.cos(phi5)
    p5y = p5 * np.sin(theta5) * np.sin(phi5)
    p5z = p5 * np.cos(theta5)

    p6x = p6 * np.sin(theta6) * np.cos(phi6)
    p6y = p6 * np.sin(theta6) * np.sin(phi6)
    p6z = p6 * np.cos(theta6)

    E1 = np.sqrt(p1**2 + mTop**2)
    E2 = np.sqrt(p2**2 + mTop**2)
    E4 = p4
    E5 = p5
    E6 = p6

    t1    = np.array([E1, p1x, p1y, p1z])
    tbar2 = np.array([E2, p2x, p2y, p2z])
    h3    = np.array([mHiggs, 0, 0, 0])
    g4    = np.array([E4, p4x, p4y, p4z])
    g5    = np.array([E5, p5x, p5y, p5z])
    g6    = np.array([E6, p6x, p6y, p6z])

    momenta = [t1, tbar2, h3, g4, g5, g6]
    
    return momenta

def tthggggMomenta(angles, p1, p2):
    phi1   = angles[0]
    theta1 = angles[1]
    phi2   = angles[2]
    theta2 = angles[3]
    phi4   = angles[4]
    theta4 = angles[5]
    phi5   = angles[6]
    theta5 = angles[7]
    phi6   = angles[8]
    theta6 = angles[9]
    phi7   = angles[10]
    theta7 = angles[11]

    def F(x):
        p4 = x[0]
        p5 = x[1]
        p6 = x[2]
        p7 = x[3]
        F = np.empty(4)
        F[0] = (np.emath.sqrt(p1**2 + mTop**2) + np.emath.sqrt(p2**2 + mTop**2)
                + p4 + p5 + p6 + p7 + mHiggs)
        F[1] = (p1 * np.sin(theta1) * np.cos(phi1)
                + p2 * np.sin(theta2) * np.cos(phi2)
                + p4 * np.sin(theta4) * np.cos(phi4)
                + p5 * np.sin(theta5) * np.cos(phi5)
                + p6 * np.sin(theta6) * np.cos(phi6)
                + p7 * np.sin(theta7) * np.cos(phi7))
        F[2] = (p1 * np.sin(theta1) * np.sin(phi1)
                + p2 * np.sin(theta2) * np.sin(phi2)
                + p4 * np.sin(theta4) * np.sin(phi4)
                + p5 * np.sin(theta5) * np.sin(phi5)
                + p6 * np.sin(theta6) * np.sin(phi6)
                + p7 * np.sin(theta7) * np.sin(phi7))
        F[3] = (p1 * np.cos(theta1)
                + p2 * np.cos(theta2)
                + p4 * np.cos(theta4)
                + p5 * np.cos(theta5)
                + p6 * np.cos(theta6)
                + p7 * np.cos(theta7))
        return F

    guess = np.array([-p1, -p1, -p1, -p1])
    p_opt = opt.newton_krylov(F, guess)

    p4 = p_opt[0]
    p5 = p_opt[1]
    p6 = p_opt[2]
    p7 = p_opt[3]

    p1x = p1 * np.sin(theta1) * np.cos(phi1)
    p1y = p1 * np.sin(theta1) * np.sin(phi1)
    p1z = p1 * np.cos(theta1)

    p2x = p2 * np.sin(theta2) * np.cos(phi2)
    p2y = p2 * np.sin(theta2) * np.sin(phi2)
    p2z = p2 * np.cos(theta2)

    p4x = p4 * np.sin(theta4) * np.cos(phi4)
    p4y = p4 * np.sin(theta4) * np.sin(phi4)
    p4z = p4 * np.cos(theta4)

    p5x = p5 * np.sin(theta5) * np.cos(phi5)
    p5y = p5 * np.sin(theta5) * np.sin(phi5)
    p5z = p5 * np.cos(theta5)

    p6x = p6 * np.sin(theta6) * np.cos(phi6)
    p6y = p6 * np.sin(theta6) * np.sin(phi6)
    p6z = p6 * np.cos(theta6)

    p7x = p7 * np.sin(theta7) * np.cos(phi7)
    p7y = p7 * np.sin(theta7) * np.sin(phi7)
    p7z = p7 * np.cos(theta7)

    E1 = np.sqrt(p1**2 + mTop**2)
    E2 = np.sqrt(p2**2 + mTop**2)
    E4 = p4
    E5 = p5
    E6 = p6
    E7 = p7

    t1    = np.array([E1, p1x, p1y, p1z])
    tbar2 = np.array([E2, p2x, p2y, p2z])
    h3    = np.array([mHiggs, 0, 0, 0])
    g4    = np.array([E4, p4x, p4y, p4z])
    g5    = np.array([E5, p5x, p5y, p5z])
    g6    = np.array([E6, p6x, p6y, p6z])
    g7    = np.array([E7, p7x, p7y, p7z])

    momenta = [t1, tbar2, h3, g4, g5, g6, g7]
    return momenta

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
            elif len(P) == 3:
                momenta = P[0].momentum() + P[1].momentum()
                z = ((  pi.abra @ momenta @ pi.sket
                    + P[-2].abra @ P[0].momentum() @ P[-2].sket)
                    / (pi.abra @ momenta @ pj.sket))
            else:
                momenta1 = np.array([p.momentum() for p in P[:-1]])
                momenta2 = np.array([p.momentum() for p in P[:-2]])
                sum1 = np.array([[0j, 0j], [0j, 0j]])
                sum2 = np.array([[0j, 0j], [0j, 0j]])
                for m in momenta1:
                    sum1 += m
                for m in momenta2:
                    sum2 += m
                z = ((  pi.abra @ sum1 @ pi.sket
                    + P[-2].abra @ sum2 @ P[-2].sket
                    + P[-3].abra @ P[0].momentum() @ P[-3].sket)
                    / (pi.abra @ sum1 @ pj.sket))
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
            elif len(P) == 3:
                momenta = P[0].momentum() + P[1].momentum()
                z = - ((  pi.abra @ momenta @ pi.sket
                    + P[-2].abra @ P[0].momentum() @ P[-2].sket)
                    / (pj.abra @ momenta @ pi.sket))
            else:
                momenta1 = P[0].momentum() + P[1].momentum() + P[2].momentum()
                momenta2 = P[0].momentum() + P[1].momentum()
                z = - ((  pi.abra @ momenta1 @ pi.sket
                    + P[-2].abra @ momenta2 @ P[-2].sket
                    + P[-3].abra @ P[0].momentum() @ P[-3].sket)
                    / (pj.abra @ momenta1 @ pi.sket))
            z = z[0, 0]
            hati.aket = pi.aket + z * pj.aket
            hati.abra = pi.abra + z * pj.abra
            hati.vector = np.array([(hati.abra @ sigma @ hati.sket)[0, 0] / 2 for sigma in pauli])
            hatj.sbra = pj.sbra - z * pi.sbra
            hatj.sket = pj.sket - z * pi.sket
            hatj.vector = np.array([(hatj.abra @ sigma @ hatj.sket)[0, 0] / 2 for sigma in pauli])
    return hati, hatj

def onShell_massive(pi, pj, P, hcase):
    z = 0
    hati = copy.deepcopy(pi)
    hatj = copy.deepcopy(pj)
    match hcase[-1]:
        case '+':
            if len(P) == 2:
                z = (pi.abra @ P[0].momentum() @ pi.sket) / (pi.abra @ P[0].momentum() @ pj.sket)
            elif len(P) == 3:
                momenta = P[0].momentum() + P[1].momentum()
                z = pi.mass * (((P[-2].abra @ P[0].momentum() @ P[-2].sket)
                       + (pj.abra @ momenta @ pj.sket))
                       / (pj.abra @ momenta @ pi.sketabra() @ pj.aket))
            else:
                momenta1 = np.array([p.momentum() for p in P[:-1]])
                momenta2 = np.array([p.momentum() for p in P[:-2]])
                sum1 = np.array([[0j, 0j], [0j, 0j]])
                sum2 = np.array([[0j, 0j], [0j, 0j]])
                for m in momenta1:
                    sum1 += m
                for m in momenta2:
                    sum2 += m
                z = ((  pi.abra @ sum1 @ pi.sket
                    + P[-2].abra @ sum2 @ P[-2].sket
                    + P[-3].abra @ P[0].momentum() @ P[-3].sket)
                    / (pi.abra @ sum1 @ pj.sket))
            z = z[0, 0]
            hati.abra = pi.abra - (z / pi.mass) * (abraket(pi, pj) @ pj.abra)
            hati.aket = pi.aket - (z / pi.mass) * (pj.aket @ abraket(pj, pi))
            hati.vector = np.array([np.trace((hati.aket @ epsLow @ hati.sbra) @ sigma) / 2
                                    for sigma in pauliBar])
            hatj.sbra = pj.sbra + (z / pi.mass) * (pj.abra @ pi.momentum())
            hatj.sket = pj.sket - (z / pi.mass) * (pi.sketabra() @ pj.aket)
            hatj.vector = np.array([(hatj.abra @ sigma @ hatj.sket)[0, 0] / 2 for sigma in pauli])
        case '-':
            if len(P) == 2:
                z = - (pi.abra @ P[0].momentum() @ pi.sket) / (pj.abra @ P[0].momentum() @ pi.sket)
            elif len(P) == 3:
                momenta = P[0].momentum() + P[1].momentum()
                z = - ((  pi.abra @ momenta @ pi.sket
                    + P[-2].abra @ P[0].momentum() @ P[-2].sket)
                    / (pj.abra @ momenta @ pi.sket))
            else:
                momenta1 = P[0].momentum() + P[1].momentum() + P[2].momentum()
                momenta2 = P[0].momentum() + P[1].momentum()
                z = - ((  pi.abra @ momenta1 @ pi.sket
                    + P[-2].abra @ momenta2 @ P[-2].sket
                    + P[-3].abra @ P[0].momentum() @ P[-3].sket)
                    / (pj.abra @ momenta1 @ pi.sket))
            z = z[0, 0]
            hati.aket = pi.aket + z * pj.aket
            hati.abra = pi.abra + z * pj.abra
            hati.vector = np.array([(hati.abra @ sigma @ hati.sket)[0, 0] / 2 for sigma in pauli])
            hatj.sbra = pj.sbra - z * pi.sbra
            hatj.sket = pj.sket - z * pi.sket
            hatj.vector = np.array([(hatj.abra @ sigma @ hatj.sket)[0, 0] / 2 for sigma in pauli])
    return hati, hatj