import amplitudes.helpers as hp
import numpy as np
import copy

def ggg(g1, g2, g3, hcase):
    '''from Campbell2023'''
    match hcase:
        case ['-', '-', '+']:
            if abs(hp.abraket(g1, g2)) <= 1e-10:
                return 0
            return hp.abraket(g1, g2)**3 / hp.abraket(g2, g3) / hp.abraket(g3, g1)
        case ['+', '+', '-']:
            if abs(hp.sbraket(g1, g2)) <= 1e-10:
                return 0
            return hp.sbraket(g1, g2)**3 / hp.sbraket(g2, g3) / hp.sbraket(g3, g1)
        case ['-', '+', '-']:
            if abs(hp.abraket(g3, g1)) <= 1e-10:
                return 0
            return hp.abraket(g3, g1)**3 / hp.abraket(g1, g2) / hp.abraket(g2, g3)
        case ['+', '-', '+']:
            if abs(hp.sbraket(g3, g1)) <= 1e-10:
                return 0
            return hp.sbraket(g3, g1)**3 / hp.sbraket(g1, g2) / hp.sbraket(g2, g3)
        case ['+', '-', '-']:
            if abs(hp.abraket(g2, g3)) <= 1e-10:
                return 0
            return hp.abraket(g2, g3)**3 / hp.abraket(g1, g2) / hp.abraket(g3, g1)
        case ['-', '+', '+']:
            if abs(hp.sbraket(g2, g3)) <= 1e-10:
                return 0
            return hp.sbraket(g2, g3)**3 / hp.sbraket(g1, g2) / hp.sbraket(g3, g1)
    return 0

def gqq(g1, q2, qbar3, hcase):
    '''from Arkani-Hamed2021'''
    match hcase:
        case ['+', '-', '+']:
            if abs(hp.sbraket(qbar3, g1)) <= 1e-10:
                return 0
            return (-1j) * hp.sbraket(qbar3, g1)**2 / hp.sbraket(q2, qbar3)
        case ['+', '+', '-']:
            if abs(hp.sbraket(g1, q2)) <= 1e-10:
                return 0
            return (-1j) * hp.sbraket(g1, q2)**2 / hp.sbraket(q2 ,qbar3)
        case ['-', '-', '+']:
            if abs(hp.abraket(g1, q2)) <= 1e-10:
                return 0
            return (-1j) * hp.abraket(g1, q2)**2 / hp.abraket(q2 ,qbar3)
        case ['-', '+', '-']:
            if abs(hp.abraket(qbar3, g1)) <= 1e-10:
                return 0
            return (-1j) * hp.abraket(qbar3, g1)**2 / hp.abraket(q2, qbar3)
    return 0

def ttg(t1, tbar2, g3, hcase, ref):
    match hcase:
        case '+':
            amp = (ref.abra @ t1.momentum() @ g3.sket) * hp.abraket(t1, tbar2) / (t1.mass * hp.abraket(ref, g3))
            return amp / 1j
        case '-':
            amp = (g3.abra @ t1.momentum() @ ref.sket) * hp.sbraket(t1, tbar2) / (t1.mass * hp.sbraket(g3, ref))
            return amp / 1j
    raise ValueError('missing gluon helicity')

def tth(t1, tbar2, h3):
    amp = hp.abraket(t1, tbar2) + hp.sbraket(t1, tbar2)
    return amp

def gggg(g1, g2, g3, g4, hcase):
    hcase = np.array(hcase)
    momenta = np.array([g1, g2, g3, g4])
    negatives = np.array(momenta[hcase == '-'])
    # if not MHV, amplitude is zero
    if len(negatives) != 2:
        return 0
    else:
        return hp.PT4(g1, g2, g3, g4, negatives)

def gqqg(g1, q2, qbar3, g4, hcase):
    '''from Elvang and Huang'''
    match hcase:
        case ['+', '+', '-', '-']:
            amp = hp.abraket(g4, qbar3)**2 / (hp.abraket(q2, g1) * hp.abraket(g1, qbar3))
            return amp
        case ['-', '+', '-', '+']:
            amp = hp.abraket(g1, qbar3)**2 / (hp.abraket(q2, g4) * hp.abraket(g4, qbar3))
            return amp
        case ['+', '-', '+', '-']:
            amp = hp.sbraket(g1, qbar3)**2 / (hp.sbraket(q2, g4) * hp.sbraket(g4, qbar3))
            return amp
        case ['-', '-', '+', '+']:
            amp = hp.sbraket(g4, qbar3)**2 / (hp.sbraket(q2, g1) * hp.sbraket(g1, qbar3))
            return amp
    return 0

def qqqq(q1, qbar2, q3, qbar4, hcase):
    p34 = q3.vector + qbar4.vector
    s34 = hp.minkowski(p34)
    match hcase:
        case ['+', '-', '+', '-']:
            return hp.sbraket(q1, q3) * hp.abraket(qbar4, qbar2) / s34
        case ['+', '-', '-', '+']:
            return hp.sbraket(q1, qbar4) * hp.abraket(q3, qbar2) / s34
        case ['-', '+', '+', '-']:
            return hp.abraket(q1, qbar4) * hp.sbraket(q3, qbar2) / s34
        case ['-', '+', '-', '+']:
            return hp.abraket(q1, q3) * hp.sbraket(qbar4, qbar2) / s34
    return 0

def ttgg(t1, tbar2, g3, g4, hcase):
    '''from Campbell2023'''
    p34 = g3.vector + g4.vector
    s34 = hp.minkowski(p34)
    m = t1.mass
    match hcase:
        case ['+', '+']:
            amp = (m * hp.sbraket(g3, g4) * hp.abraket(t1, tbar2)
                   / (hp.abraket(g3, g4) * (g3.abra @ t1.momentum() @ g3.sket)))
            return amp / (-1j)
        case ['-', '-']:
            amp = (m * hp.abraket(g3, g4) * hp.sbraket(t1, tbar2)
                   / (hp.sbraket(g3, g4) * (g3.abra @ t1.momentum() @ g3.sket)))
            return amp / (-1j)
        case ['+', '-']:
            amp = ((g4.abra @ t1.momentum() @ g3.sket)
                   * (hp.sbraket(t1, g3) @ hp.abraket(g4, tbar2) + hp.abraket(t1, g4) @ hp.sbraket(g3, tbar2))
                   / ((g3.abra @ t1.momentum() @ g3.sket) * s34))
            return amp / (-1j)
        case ['-', '+']:
            amp = ((g3.abra @ t1.momentum() @ g4.sket)
                   * (hp.abraket(t1, g3) @ hp.sbraket(g4, tbar2) + hp.sbraket(t1, g4) @ hp.abraket(g3, tbar2))
                   / ((g3.abra @ t1.momentum() @ g3.sket) * s34))
            return amp / (-1j)
    raise ValueError('missing gluon helicities')

def ttqq(t1, tbar2, q3, qbar4, hcase):
    '''from Campbell2023'''
    p34 = q3.vector + qbar4.vector
    s34 = hp.minkowski(p34)
    match hcase:
        case ['+', '-']:
            amp = (hp.abraket(t1, qbar4) @ hp.sbraket(q3, tbar2) + hp.sbraket(t1, q3) @ hp.abraket(qbar4, tbar2)) / s34
            return amp / (-1j)
        case ['-', '+']:
            amp = (hp.abraket(t1, q3) @ hp.sbraket(qbar4, tbar2) + hp.sbraket(t1, qbar4) @ hp.abraket(q3, tbar2)) / s34
            return amp / (-1j)
    return np.array([[0, 0], [0, 0]])

def tthg(t1, tbar2, h3, g4, hcase, ref):
    if abs(hp.abraket(g4, ref)) <= 1e-10 and abs(hp.sbraket(g4, ref)) <= 1e-10:
        raise ValueError('please pick a different reference momentum ref ~/~ g4')
    match hcase:
        case '+':
            d1 = ((hp.abraket(t1, ref)
                  @ (2 * t1.mass * hp.sbraket(g4, tbar2) + (g4.sbra @ h3.sketabra() @ tbar2.aket))
                  + hp.sbraket(t1, g4)
                  @ (2 * t1.mass * hp.abraket(ref, tbar2) + (ref.abra @ h3.momentum() @ tbar2.sket)))
                  / (g4.abra @ t1.momentum() @ g4.sket))
            d2 = (((2 * t1.mass * hp.abraket(t1, ref) - (t1.sbra @ h3.sketabra() @ ref.aket))
                   @ hp.sbraket(g4, tbar2)
                   + (2 * t1.mass * hp.sbraket(t1, g4) - (t1.abra @ h3.momentum() @ g4.sket))
                   @ hp.abraket(ref, tbar2))
                  / (g4.abra @ tbar2.momentum() @ g4.sket))
            return (1 / hp.abraket(g4, ref)) * (d1 + d2)
        case '-':
            d1 = ((hp.sbraket(t1, ref)
                  @ (2 * t1.mass * hp.abraket(g4, tbar2) + (g4.abra @ h3.momentum() @ tbar2.sket))
                  + hp.abraket(t1, g4)
                  @ (2 * t1.mass * hp.sbraket(ref, tbar2) + (ref.sbra @ h3.sketabra() @ tbar2.aket)))
                  / (g4.abra @ t1.momentum() @ g4.sket))
            d2 = (((2 * t1.mass * hp.sbraket(t1, ref) - (t1.abra @ h3.momentum() @ ref.sket))
                   @ hp.abraket(g4, tbar2)
                   + (2 * t1.mass * hp.abraket(t1, g4) - (t1.sbra @ h3.sketabra() @ g4.aket))
                   @ hp.sbraket(ref, tbar2))
                  / (g4.abra @ tbar2.momentum() @ g4.sket))
            return (1 / hp.sbraket(ref, g4)) * (d1 + d2)
    raise ValueError('missing gluon helicity')

def ttggg(t1, tbar2, g3, g4, g5, hcase):
    P = [t1, g5]
    p15 = t1.vector + g5.vector
    s15 = hp.minkowski(p15)
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='left')
    hatp15 = hp.massive((t1.vector + hat5.vector))
    negp15 = hp.massive((-t1.vector - hat5.vector))
    d1 = (ttg(t1, negp15, hat5, hcase[-1], ref=g4)
          @ hp.epsLow @ ttgg(hatp15, tbar2, g3, hat4, hcase[:-1])) / (s15 - t1.mass**2)
    
    P = [g3, g4]
    p34 = g3.vector + g4.vector
    s34 = hp.minkowski(p34)
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='right')
    hatp34 = hp.massless((g3.vector + hat4.vector))
    negp34 = hp.massless((-g3.vector - hat4.vector))
    d2a = (ttgg(t1, tbar2, hatp34, hat5, ['+'] + list(hcase[-1]))
           * ggg(negp34, g3, hat4, ['-'] + list(hcase[:-1]))) / s34
    d2b = (ttgg(t1, tbar2, hatp34, hat5, ['-'] + list(hcase[-1]))
           * ggg(negp34, g3, hat4, ['+'] + list(hcase[:-1]))) / s34
    
    amp = d1 + d2a + d2b
    return amp

def ttqqg(t1, tbar2, q3, qbar4, g5, hcase):
    P = [t1, g5]
    p15 = t1.vector + g5.vector
    s15 = hp.minkowski(p15)
    hat4, hat5 = hp.onShell(qbar4, g5, P, hcase, side='left')
    hatp15 = hp.massive((t1.vector + hat5.vector))
    negp15 = hp.massive((-t1.vector - hat5.vector))
    d1 = -1j * (ttg(t1, negp15, hat5, hcase[-1], ref=qbar4)
          @ hp.epsLow @ ttqq(hatp15, tbar2, q3, hat4, hcase[:-1])) / (s15 - t1.mass**2)

    P = [q3, qbar4]
    p34 = q3.vector + qbar4.vector
    s34 = hp.minkowski(p34)
    hat4, hat5 = hp.onShell(qbar4, g5, P, hcase, side='right')
    hatp34 = hp.massless((q3.vector + hat4.vector))
    negp34 = hp.massless((-q3.vector - hat4.vector))
    d2a = 1j * (ttgg(t1, tbar2, hatp34, hat5, ['+'] + list(hcase[-1]))
                * gqq(negp34, q3, hat4, ['-']+ list(hcase[:-1]))) / s34
    
    d2b = 1j * (ttgg(t1, tbar2, hatp34, hat5, ['-'] + list(hcase[-1]))
                * gqq(negp34, q3, hat4, ['+']+ list(hcase[:-1]))) / s34

    amp = d1 + d2a + d2b
    return amp

def tthgg(t1, tbar2, h3, g4, g5, hcase):
    P = [t1, g5]
    p15 = t1.vector + g5.vector
    s15 = hp.minkowski(p15)
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='left')
    hatp15 = hp.massive((t1.vector + hat5.vector))
    negp15 = hp.massive((-t1.vector - hat5.vector))
    d1 = (ttg(t1, negp15, hat5, hcase[1], ref=g4)
          @ hp.epsLow @ tthg(hatp15, tbar2, h3, hat4, hcase[0], ref=g5)) / (s15 - t1.mass**2)

    P = [tbar2, g4]
    p24 = tbar2.vector + g4.vector
    s24 = hp.minkowski(p24)
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='right')
    hatp24 = hp.massive((tbar2.vector + hat4.vector))
    negp24 = hp.massive((-tbar2.vector - hat4.vector))
    d2 = (tthg(t1, hatp24, h3, hat5, hcase[1], ref=g4)
          @ hp.epsLow @ ttg(negp24, tbar2, hat4, hcase[0], ref=g5)) / (s24 - t1.mass**2)
    
    amp = d1 + d2
    return amp

def tthgg_antiholo(t1, tbar2, h3, g4, g5, hcase):
    P = [tbar2, g4, g5]
    p245 = tbar2.vector + g4.vector + g5.vector
    s245 = hp.minkowski(p245)
    z245 = - t1.mass * (((g4.abra @ tbar2.momentum() @ g4.sket)
                       + (g5.abra @ tbar2.momentum() @ g5.sket)
                       + (g5.abra @ g4.momentum() @ g5.sket))
                       / (g5.sbra @ t1.sketabra() @ (tbar2.momentum() + g4.momentum()) @ g5.sket))
    hat1 = copy.deepcopy(t1)
    hat5 = copy.deepcopy(g5)
    hat1.sbra = t1.sbra - (z245 / t1.mass) * (hp.sbraket(t1, g5) @ g5.sbra)
    hat1.sket = t1.sket - (z245 / t1.mass) * (g5.sket @ hp.sbraket(g5, t1))
    hat1.vector = np.array([np.trace((hat1.aket @ hp.epsLow @ hat1.sbra) @ sigma) / 2
                            for sigma in hp.pauliBar])
    hat5.abra = g5.abra - (z245 / t1.mass) * (g5.sbra @ t1.sketabra())
    hat5.aket = g5.aket + (z245 / t1.mass) * (t1.momentum() @ g5.sket)
    hat5.vector = np.array([(hat5.abra @ sigma @ hat5.sket)[0, 0] / 2 for sigma in hp.pauli])
    hatp245 = hp.massive((tbar2.vector + g4.vector + hat5.vector))
    negp245 = hp.massive((- tbar2.vector - g4.vector - hat5.vector))
    d1 = (tth(hat1, hatp245, h3) @ hp.epsLow @ ttgg(negp245, tbar2, g4, hat5, hcase)) / (s245 - t1.mass**2)

    P = [g4, g5]
    p45 = g4.vector + g5.vector
    s45 = hp.minkowski(p45)
    z45 = - t1.mass * ((g5.sbra @ g4.sketabra() @ g5.aket) / (g5.sbra @ g4.sketabra() @ t1.momentum() @ g5.sket))
    hat1 = copy.deepcopy(t1)
    hat5 = copy.deepcopy(g5)
    hat1.sbra = t1.sbra - (z45 / t1.mass) * (hp.sbraket(t1, g5) @ g5.sbra)
    hat1.sket = t1.sket - (z45 / t1.mass) * (g5.sket @ hp.sbraket(g5, t1))
    hat1.vector = np.array([np.trace((hat1.aket @ hp.epsLow @ hat1.sbra) @ sigma) / 2
                            for sigma in hp.pauliBar])
    hat5.abra = g5.abra - (z45 / t1.mass) * (g5.sbra @ t1.sketabra())
    hat5.aket = g5.aket + (z45 / t1.mass) * (t1.momentum() @ g5.sket)
    hat5.vector = np.array([(hat5.abra @ sigma @ hat5.sket)[0, 0] / 2 for sigma in hp.pauli])
    # print(f'p2.p45^ = {hp.dot(tbar2.vector, g4.vector + hat5.vector)}')
    # print(f'p1^3.p45^ = {hp.dot(hat1.vector + h3.vector, g4.vector + hat5.vector)}')
    hatp45 = hp.massless(g4.vector + hat5.vector)
    negp45 = hp.massless(-g4.vector - hat5.vector)
    d2a = (tthg(hat1, tbar2, h3, hatp45, '+', ref=g5) * ggg(negp45, g4, hat5, ['-'] + list(hcase))) / s45
    d2b = (tthg(hat1, tbar2, h3, hatp45, '-', ref=g5) * ggg(negp45, g4, hat5, ['+'] + list(hcase))) / s45

    # print(f'<P^|1^|P^] = {hatp45.abra @ hat1.momentum() @ hatp45.sket}')
    # print(f'p1^.p4 = {hp.dot(hat1.vector, g4.vector)}')
    # print(f'p1^.p5^ = {hp.dot(hat1.vector, hat5.vector)}')
    print(f'p2 = {tbar2.vector}')
    print(f'p45^ = {hatp45.vector}')
    print(f'p4 = {g4.vector}')
    print(f'p5^ = {hat5.vector}')
    print(f'p2.p45^ = {hp.dot(tbar2.vector, hatp45.vector)}')
    print(f'p1^3.p45^ = {hp.dot(hat1.vector + h3.vector, hatp45.vector)}')
    print(f'p13.p45 = {hp.dot(t1.vector + h3.vector, p45)}')
    print(f'p2.p45 = {hp.dot(tbar2.vector, p45)}')

    amp = d1 + d2a + d2b
    return amp

def tthgg_massive(t1, tbar2, h3, g4, g5, hcase):
    P = [tbar2, g4, g5]
    p245 = tbar2.vector + g4.vector + g5.vector
    s245 = hp.minkowski(p245)
    z245 = t1.mass * (((g4.abra @ tbar2.momentum() @ g4.sket)
                       + (g5.abra @ tbar2.momentum() @ g5.sket)
                       + (g5.abra @ g4.momentum() @ g5.sket))
                       / (g5.abra @ (g4.momentum() + tbar2.momentum()) @ t1.sketabra() @ g5.aket))
    hat1 = copy.deepcopy(t1)
    hat5 = copy.deepcopy(g5)
    hat1.abra = t1.abra - (z245 / t1.mass) * (hp.abraket(t1, g5) @ g5.abra)
    hat1.aket = t1.aket - (z245 / t1.mass) * (g5.aket @ hp.abraket(g5, t1))
    hat1.vector = np.array([np.trace((hat1.aket @ hp.epsLow @ hat1.sbra) @ sigma) / 2
                            for sigma in hp.pauliBar])
    hat5.sbra = g5.sbra + (z245 / t1.mass) * (g5.abra @ t1.momentum())
    hat5.sket = g5.sket - (z245 / t1.mass) * (t1.sketabra() @ g5.aket)
    hat5.vector = np.array([(hat5.abra @ sigma @ hat5.sket)[0, 0] / 2 for sigma in hp.pauli])
    hatp245 = hp.massive((tbar2.vector + g4.vector + hat5.vector))
    negp245 = hp.massive((- tbar2.vector - g4.vector - hat5.vector))
    d1 = (tth(hat1, hatp245, h3) @ hp.epsLow @ ttgg(negp245, tbar2, g4, hat5, hcase)) / (s245 - t1.mass**2)

    print(hp.sbraket(g4, hat5))
    print(ttgg(negp245, tbar2, g4, hat5, hcase))


    P = [g4, g5]
    p45 = g4.vector + g5.vector
    s45 = hp.minkowski(p45)
    z45 = t1.mass * (g5.abra @ g4.momentum() @ g5.sket) / (g5.abra @ g4.momentum() @ t1.sketabra() @ g5.aket)
    hat1 = copy.deepcopy(t1)
    hat5 = copy.deepcopy(g5)
    hat1.abra = t1.abra - (z45 / t1.mass) * (hp.abraket(t1, g5) @ g5.abra)
    hat1.aket = t1.aket - (z45 / t1.mass) * (g5.aket @ hp.abraket(g5, t1))
    hat1.vector = np.array([np.trace((hat1.aket @ hp.epsLow @ hat1.sbra) @ sigma) / 2
                            for sigma in hp.pauliBar])
    hat5.sbra = g5.sbra + (z45 / t1.mass) * (g5.abra @ t1.momentum())
    hat5.sket = g5.sket - (z45 / t1.mass) * (t1.sketabra() @ g5.aket)
    hat5.vector = np.array([(hat5.abra @ sigma @ hat5.sket)[0, 0] / 2 for sigma in hp.pauli])
    hatp45 = hp.massless(g4.vector + hat5.vector)
    negp45 = hp.massless(-g4.vector - hat5.vector)
    d2a = (tthg(hat1, tbar2, h3, hatp45, '+', ref=g5) * ggg(negp45, g4, hat5, ['-'] + list(hcase))) / s45
    d2b = (tthg(hat1, tbar2, h3, hatp45, '-', ref=g5) * ggg(negp45, g4, hat5, ['+'] + list(hcase))) / s45

    amp = d1 + d2a + d2b
    return amp

def tthqq(t1, tbar2, h3, q4, qbar5, hcase):
    m = t1.mass
    p45 = q4.vector + qbar5.vector
    s45 = hp.minkowski(p45)
    match hcase:
        case ['+', '-']:
            d1 = 2 * (((2 * m * hp.abraket(t1, qbar5) - (t1.sbra @ h3.sketabra() @ qbar5.aket)) @ hp.sbraket(q4, tbar2)
                      + (2 * m * hp.sbraket(t1, q4) - (t1.abra @ h3.momentum() @ q4.sket)) @ hp.abraket(qbar5, tbar2))
                      / (((q4.abra @ tbar2.momentum() @ q4.sket) + (qbar5.abra @ tbar2.momentum() @ qbar5.sket) + s45) * s45))
            d2 = 2 * ((hp.abraket(t1, qbar5) @ (2 * m * hp.sbraket(q4, tbar2) + (q4.sbra @ h3.sketabra() @ tbar2.aket))
                       + hp.sbraket(t1, q4) @ (2 * m * hp.abraket(qbar5, tbar2) + (qbar5.abra @ h3.momentum() @ tbar2.sket)))
                      / (((q4.abra @ t1.momentum() @ q4.sket) + (qbar5.abra @ t1.momentum() @ qbar5.sket) + s45) * s45))
            return d1 + d2
        case ['-', '+']:
            d1 = 2 * (((2 * m * hp.abraket(t1, q4) - (t1.sbra @ h3.sketabra() @ q4.aket)) @ hp.sbraket(qbar5, tbar2)
                      + (2 * m * hp.sbraket(t1, qbar5) - (t1.abra @ h3.momentum() @ qbar5.sket)) @ hp.abraket(q4, tbar2))
                      / (((q4.abra @ tbar2.momentum() @ q4.sket) + (qbar5.abra @ tbar2.momentum() @ qbar5.sket) + s45) * s45))
            d2 = 2 * ((hp.abraket(t1, q4) @ (2 * m * hp.sbraket(qbar5, tbar2) + (qbar5.sbra @ h3.sketabra() @ tbar2.aket))
                       + hp.sbraket(t1, qbar5) @ (2 * m * hp.abraket(q4, tbar2) + (q4.abra @ h3.momentum() @ tbar2.sket)))
                      / (((q4.abra @ t1.momentum() @ q4.sket) + (qbar5.abra @ t1.momentum() @ qbar5.sket) + s45) * s45))
            return d1 + d2
    return np.array([[0, 0], [0, 0]])

def tthggg(t1, tbar2, h3, g4, g5, g6, hcase):
    P = [t1, g6]
    p16 = t1.vector + g6.vector
    s16 = hp.minkowski(p16)
    hat5, hat6 = hp.onShell(g5, g6, P, hcase, side='left')
    hatp16 = hp.massive((t1.vector + hat6.vector))
    negp16 = hp.massive((-t1.vector - hat6.vector))
    d1 = (ttg(t1, negp16, hat6, hcase[2], ref=g5)
          @ hp.epsLow @ tthgg(hatp16, tbar2, h3, g4, hat5, hcase[:2])) / (s16 - t1.mass**2)

    P = [tbar2, g4, g5]
    p245 = tbar2.vector + g4.vector + g5.vector
    s245 = hp.minkowski(p245)
    hat5, hat6 = hp.onShell(g5, g6, P, hcase, side='right')
    hatp245 = hp.massive((tbar2.vector + g4.vector + hat5.vector))
    negp245 = hp.massive((-tbar2.vector - g4.vector - hat5.vector))
    d2 = (tthg(t1, hatp245, h3, hat6, hcase[2], g5)
          @ hp.epsLow @ ttgg(negp245, tbar2, g4, hat5, hcase[:2])) / (s245 - t1.mass**2)

    P = [g4, g5]
    p45 = g4.vector + g5.vector
    s45 = hp.minkowski(p45)
    hat5, hat6 = hp.onShell(g5, g6, P, hcase, side='right')
    hatp45 = hp.massless((g4.vector + hat5.vector))
    negp45 = hp.massless((-g4.vector - hat5.vector))
    d3a = (tthgg(t1, tbar2, h3, hatp45, hat6, ['+'] + list(hcase[2]))
           * ggg(negp45, g4, hat5, ['-'] + list(hcase[:2]))) / s45
    d3b = (tthgg(t1, tbar2, h3, hatp45, hat6, ['-'] + list(hcase[2]))
           * ggg(negp45, g4, hat5, ['+'] + list(hcase[:2]))) / s45
    
    amp = d1 + d2 + d3a + d3b
    return amp

def tthqqg(t1, tbar2, h3, q4, qbar5, g6, hcase):
    P = [t1, g6]
    p16 = t1.vector + g6.vector
    s16 = hp.minkowski(p16)
    hat5, hat6 = hp.onShell(qbar5, g6, P, hcase, side='left')
    hatp16 = hp.massive((t1.vector + hat6.vector))
    negp16 = hp.massive((-t1.vector - hat6.vector))
    d1 = (ttg(t1, negp16, hat6, hcase[2], ref=qbar5)
          @ hp.epsLow @ tthqq(hatp16, tbar2, h3, q4, hat5, hcase[:2])) / (s16 - t1.mass**2)

    P = [tbar2, q4, qbar5]
    p245 = tbar2.vector + q4.vector + qbar5.vector
    s245 = hp.minkowski(p245)
    hat5, hat6 = hp.onShell(qbar5, g6, P, hcase, side='right')
    hatp245 = hp.massive((tbar2.vector + q4.vector + hat5.vector))
    negp245 = hp.massive((-tbar2.vector - q4.vector - hat5.vector))
    d2 = (tthg(t1, hatp245, h3, hat6, hcase[2], qbar5)
          @ hp.epsLow @ ttqq(negp245, tbar2, q4, hat5, hcase[:2])) / (s245 - t1.mass**2)

    P = [q4, qbar5]
    p45 = q4.vector + qbar5.vector
    s45 = hp.minkowski(p45)
    hat5, hat6 = hp.onShell(qbar5, g6, P, hcase, side='right')
    hatp45 = hp.massless((q4.vector + hat5.vector))
    negp45 = hp.massless((-q4.vector - hat5.vector))
    d3a = (tthgg(t1, tbar2, h3, negp45, hat6, ['+'] + list(hcase[2]))
           * gqq(hatp45, q4, hat5, ['-'] + list(hcase[:2]))) / s45
    d3b = (tthgg(t1, tbar2, h3, negp45, hat6, ['-'] + list(hcase[2]))
           * gqq(hatp45, q4, hat5, ['+'] + list(hcase[:2]))) / s45

    amp = d1 + d2 + d3a + d3b
    return amp

def tthgggg(t1, tbar2, h3, g4, g5, g6, g7, hcase):
    P = [t1, g7]
    p17 = t1.vector + g7.vector
    s17 = hp.minkowski(p17)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='left')
    hatp17 = hp.massive((t1.vector + hat7.vector))
    negp17 = hp.massive((-t1.vector - hat7.vector))
    d1 = (ttg(t1, negp17, hat7, hcase[3], ref=g6)
          @ hp.epsLow @ tthggg(hatp17, tbar2, h3, g4, g5, hat6, hcase[:3])) / (s17 - t1.mass**2)

    P = [tbar2, g4, g5, g6]
    p2456 = tbar2.vector + g4.vector + g5.vector + g6.vector
    s2456 = hp.minkowski(p2456)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp2456 = hp.massive((tbar2.vector + g4.vector + g5.vector + hat6.vector))
    negp2456 = hp.massive((-tbar2.vector - g4.vector - g5.vector - hat6.vector))
    d2 = (tthg(t1, hatp2456, h3, hat7, hcase[3], g6)
          @ hp.epsLow @ ttggg(negp2456, tbar2, g4, g5, hat6, hcase[:3])) / (s2456 - t1.mass**2)

    P = [g4, g5, g6]
    p456 = g4.vector + g5.vector + g6.vector
    s456 = hp.minkowski(p456)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp456 = hp.massless((g4.vector + g5.vector + hat6.vector))
    negp456 = hp.massless((-g4.vector - g5.vector - hat6.vector))
    d3a = (tthgg(t1, tbar2, h3, hatp456, hat7, ['+'] + list(hcase[3]))
           * gggg(negp456, g4, g5, hat6, ['-'] + list(hcase[:3]))) / s456
    d3b = (tthgg(t1, tbar2, h3, hatp456, hat7, ['-'] + list(hcase[3]))
           * gggg(negp456, g4, g5, hat6, ['+'] + list(hcase[:3]))) / s456

    P = [g5, g6]
    p56 = g5.vector + g6.vector
    s56 = hp.minkowski(p56)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp56 = hp.massless((g5.vector + hat6.vector))
    negp56 = hp.massless((-g5.vector - hat6.vector))
    hcase4p7a = list(hcase[0]) + ['+'] + list(hcase[3])
    hcase4p7b = list(hcase[0]) + ['-'] + list(hcase[3])
    d4a = (tthggg(t1, tbar2, h3, g4, hatp56, hat7, hcase4p7a)
           * ggg(negp56, g5, hat6, ['-'] + list(hcase[1:3]))) / s56
    d4b = (tthggg(t1, tbar2, h3, g4, hatp56, hat7, hcase4p7b)
           * ggg(negp56, g5, hat6, ['+'] + list(hcase[1:3]))) / s56
    
    amp = d1 + d2 + d3a + d3b + d4a + d4b
    return amp

def tthqqgg(t1, tbar2, h3, q4, qbar5, g6, g7, hcase):
    P = [t1, g7]
    p17 = t1.vector + g7.vector
    s17 = hp.minkowski(p17)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='left')
    hatp17 = hp.massive((t1.vector + hat7.vector))
    negp17 = hp.massive((-t1.vector - hat7.vector))
    d1 = (ttg(t1, negp17, hat7, hcase[3], ref=g6)
          @ hp.epsLow @ tthqqg(hatp17, tbar2, h3, q4, qbar5, hat6, hcase[:3])) / (s17 - t1.mass**2)

    P = [tbar2, q4, qbar5, g6]
    p2456 = tbar2.vector + q4.vector + qbar5.vector + g6.vector
    s2456 = hp.minkowski(p2456)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp2456 = hp.massive((tbar2.vector + q4.vector + qbar5.vector + hat6.vector))
    negp2456 = hp.massive((-tbar2.vector - q4.vector - qbar5.vector - hat6.vector))
    d2 = (tthg(t1, hatp2456, h3, hat7, hcase[3], g6)
          @ hp.epsLow @ ttqqg(negp2456, tbar2, q4, qbar5, hat6, hcase[:3])) / (s2456 - t1.mass**2)

    P = [q4, qbar5, g6]
    p456 = q4.vector + qbar5.vector + g6.vector
    s456 = hp.minkowski(p456)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp456 = hp.massless((q4.vector + qbar5.vector + hat6.vector))
    negp456 = hp.massless((-q4.vector - qbar5.vector - hat6.vector))
    d3a = (tthgg(t1, tbar2, h3, hatp456, hat7, ['+'] + list(hcase[3]))
           * gqqg(negp456, q4, qbar5, hat6, ['-'] + list(hcase[:3]))) / s456
    d3b = (tthgg(t1, tbar2, h3, hatp456, hat7, ['-'] + list(hcase[3]))
           * gqqg(negp456, q4, qbar5, hat6, ['+'] + list(hcase[:3]))) / s456

    P = [qbar5, g6]
    p56 = qbar5.vector + g6.vector
    s56 = hp.minkowski(p56)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp56 = hp.massless((qbar5.vector + hat6.vector))
    negp56 = hp.massless((-qbar5.vector - hat6.vector))
    hcase4p7a = list(hcase[0]) + ['+'] + list(hcase[3])
    hcase4p7b = list(hcase[0]) + ['-'] + list(hcase[3])
    d4a = (tthggg(t1, tbar2, h3, q4, hatp56, hat7, hcase4p7a)
           * gqq(hat6, negp56, qbar5, ['-'] + list(hcase[1:3]))) / s56
    d4b = (tthggg(t1, tbar2, h3, q4, hatp56, hat7, hcase4p7b)
           * gqq(hat6, negp56, qbar5, ['+'] + list(hcase[1:3]))) / s56
    amp = d1 + d2 + d3a + d3b + d4a + d4b
    return amp

def tthqqqq(t1, tbar2, h3, q4, qbar5, q6, qbar7, hcase):
    P = [q4, qbar5, q6]
    p456 = q4.vector + qbar5.vector + q6.vector
    s456 = hp.minkowski(p456)
    hat6, hat7 = hp.onShell(q6, qbar7, P, hcase, side='right')
    hatp456 = hp.massless((q4.vector + qbar5.vector + hat6.vector))
    negp456 = hp.massless((-q4.vector - qbar5.vector - hat6.vector))
    d1a = (tthqq(t1, tbar2, h3, hatp456, hat7, ['+'] + list(hcase[-1]))
           * qqqq(q4, qbar5, hat6, negp456, ['-'] + list(hcase[:-1])) / s456)
    d1b = (tthqq(t1, tbar2, h3, hatp456, hat7, ['-'] + list(hcase[-1]))
           * qqqq(q4, qbar5, hat6, negp456, ['+'] + list(hcase[:-1])) / s456)
    amp = d1a + d1b
    return amp
    