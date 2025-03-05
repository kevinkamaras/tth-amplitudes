import amplitudes.helpers as hp
import numpy as np

def tthg(t1, tbar2, h3, g4, hcase, ref):
    match hcase:
        case '+':
            d1 = (hp.abraket(t1, ref) @ (g4.sbra @ t1.sketabra @ tbar2.aket)
                  - t1.mass * (hp.abraket(t1, ref) @ hp.sbraket(g4, tbar2))
                  + hp.sbraket(t1, g4) @ (ref.abra @ (t1.momentum() + g4.momentum()) @ tbar2.sket)
                  - t1.mass * (hp.sbraket(t1, g4) @ hp.abraket(ref, tbar2))) / (g4.abra @ t1.momentum() @ g4.sket)
            d2 = ((t1.abra @ (t1.momentum() + h3.momentum()) @ g4.sket) @ hp.abraket(ref, tbar2)
                  - t1.mass * (hp.abraket(t1, ref) @ hp.sbraket(g4, tbar2))
                  + (t1.sbra @ (t1.sketabra + h3.sketabra) @ ref.aket) @ hp.sbraket(g4, tbar2)
                  - t1.mass * (hp.sbraket(t1, g4) @ hp.abraket(ref, tbar2))) / (g4.abra @ tbar2.momentum() @ g4.sket)
            return (np.sqrt(2) / hp.abraket(ref, g4)) * (d1 + d2)
        case '-':
            d1 = (hp.abraket(t1, ref) @ (g4.sbra @ h3.momentum() @ tbar2.aket)
                + hp.sbraket(t1, g4) @ (ref.abra @ h3.momentum() @ tbar2.sket)) / (g4.abra @ t1.momentum() @ g4.sket)
            d2 = ((t1.abra @ h3.momentum() @ g4.sket) @ hp.abraket(ref, tbar2)
                + (t1.sbra @ h3.momentum() @ ref.aket) @ hp.sbraket(g4, tbar2)) / (g4.abra @ tbar2.momentum() @ g4.sket)
            return (np.sqrt(2) / hp.abraket(ref, g4)) * (d1 + d2)
    raise ValueError('missing gluon helicity')

def ttggg(t1, tbar2, g3, g4, g5, hcase):
    P = [t1, g5]
    p15 = t1.vector + g5.vector
    s15 = hp.minkowski(p15)
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='left')
    hatp15 = hp.massive(*(t1.vector + hat5.vector))
    negp15 = hp.massive(*(-t1.vector - hat5.vector))
    d1 = (hp.ttg(t1, hatp15, hat5, hcase[-1], ref=g4)
          @ hp.ttgg(negp15, tbar2, g3, hat4, hcase[:-1])) / (s15 - t1.mass**2)

    P = [g3, g4]
    p34 = g3.vector + g4.vector
    s34 = hp.minkowski(p34)
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='right')
    hatp34 = hp.massless(*(g3.vector + hat4.vector))
    negp34 = hp.massless(*(-g3.vector - hat4.vector))
    d2a = (hp.ttgg(t1, tbar2, negp34, hat5, ['+'] + list(hcase[1]))
           * hp.ggg(hatp34, g3, hat4, ['-'] + list(hcase[0]))) / s34
    d2b = (hp.ttgg(t1, tbar2, negp34, hat5, ['-'] + list(hcase[1]))
           * hp.ggg(hatp34, g3, hat4, ['+'] + list(hcase[0]))) / s34
    
    amp = d1 + d2a + d2b
    return amp

def ttqqg(t1, tbar2, q3, qbar4, g5, hcase):
    P = [t1, g5]
    p15 = t1.vector + g5.vector
    s15 = hp.minkowski(p15)
    hat4, hat5, z15 = hp.onShell(qbar4, g5, P, hcase, side='left')
    hatp15 = hp.massive(*(t1.vector + hat5.vector))
    negp15 = hp.massive(*(-t1.vector - hat5.vector))
    d1 = -1j * (hp.ttg(t1, negp15, hat5, hcase[-1], ref=qbar4)
          @ hp.epsLow @ hp.ttqq(hatp15, tbar2, q3, hat4, hcase[:-1])) / (s15 - t1.mass**2)

    P = [q3, qbar4]
    p34 = q3.vector + qbar4.vector
    s34 = hp.minkowski(p34)
    hat4, hat5, z34 = hp.onShell(qbar4, g5, P, hcase, side='right')
    hatp34 = hp.massless(*(q3.vector + hat4.vector))
    negp34 = hp.massless(*(-q3.vector - hat4.vector))
    d2a = 1j * (hp.ttgg(t1, tbar2, hatp34, hat5, ['+'] + list(hcase[-1]))
           * hp.gqq(negp34, q3, hat4, ['-']+ list(hcase[:-1]), ref=g5)) / s34
    
    d2b = (hp.ttgg(t1, tbar2, hatp34, hat5, ['-'] + list(hcase[-1]))
           * hp.gqq(negp34, q3, hat4, ['+']+ list(hcase[:-1]), ref=g5)) / s34
    
    amp = d1 + d2a + d2b
    return amp

def tthgg(t1, tbar2, h3, g4, g5, hcase):
    P = [t1, g5]
    p15 = t1.vector + g5.vector
    s15 = hp.minkowski(p15)
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='left')
    hatp15 = hp.massive(*(t1.vector + hat5.vector))
    negp15 = hp.massive(*(-t1.vector - hat5.vector))
    d1 = (hp.ttg(t1, hatp15, hat5, hcase[1], ref=g4)
          @ tthg(negp15, tbar2, h3, hat4, hcase[0], g5)) / (s15 - t1.mass**2)

    P = [tbar2, g4]
    p24 = tbar2.vector + g4.vector
    s24 = hp.minkowski(p24)
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='right')
    hatp24 = hp.massive(*(tbar2.vector + hat4.vector))
    negp24 = hp.massive(*(-tbar2.vector - hat4.vector))
    d2 = (tthg(t1, negp24, h3, hat5, hcase[1], g4)
          @ hp.ttg(hatp24, tbar2, hat4, hcase[0], ref=g5)) / (s24 - t1.mass**2)
    
    amp = d1 + d2
    return amp

def tthqq(t1, tbar2, h3, q4, qbar5, hcase):
    m = t1.mass
    p45 = q4.vector + qbar5.vector
    s45 = hp.minkowski(p45)
    match hcase:
        case ['+', '-']:
            d1 = 2 * (((2 * m * hp.abraket(t1, qbar5) - (t1.sbra @ h3.sketabra @ qbar5.aket)) @ hp.sbraket(q4, tbar2)
                      + (2 * m * hp.sbraket(t1, q4) - (t1.abra @ h3.momentum() @ q4.sket)) @ hp.sbraket(qbar5, tbar2))
                      / (((q4.abra @ tbar2.momentum() @ q4.sket) + (qbar5.abra @ tbar2.momentum() @ qbar5.sket) + s45) * s45))
            d2 = 2 * ((hp.abraket(t1, qbar5) @ (2 * m * hp.sbraket(q4, tbar2) + (q4.sbra @ h3.sketabra @ tbar2.aket))
                       + hp.sbraket(t1, q4) @ (2 * m * hp.abraket(qbar5, tbar2) + (qbar5.abra @ h3.momentum() @ tbar2.sket)))
                      / (((q4.abra @ t1.momentum() @ q4.sket) + (qbar5.abra @ t1.momentum() @ qbar5.sket) + s45) * s45))
            return d1 + d2
        case ['-', '+']:
            d1 = 2 * (((2 * m * hp.abraket(t1, q4) - (t1.sbra @ h3.sketabra @ q4.aket)) @ hp.sbraket(qbar5, tbar2)
                      + (2 * m * hp.sbraket(t1, qbar5) - (t1.abra @ h3.momentum() @ qbar5.sket)) @ hp.sbraket(q4, tbar2))
                      / (((q4.abra @ tbar2.momentum() @ q4.sket) + (qbar5.abra @ tbar2.momentum() @ qbar5.sket) + s45) * s45))
            d2 = 2 * ((hp.abraket(t1, q4) @ (2 * m * hp.sbraket(qbar5, tbar2) + (qbar5.sbra @ h3.sketabra @ tbar2.aket))
                       + hp.sbraket(t1, qbar5) @ (2 * m * hp.abraket(q4, tbar2) + (q4.abra @ h3.momentum() @ tbar2.sket)))
                      / (((q4.abra @ t1.momentum() @ q4.sket) + (qbar5.abra @ t1.momentum() @ qbar5.sket) + s45) * s45))
            return d1 + d2
    return 0

def tthggg(t1, tbar2, h3, g4, g5, g6, hcase):
    P = [t1, g6]
    p16 = t1.vector + g6.vector
    s16 = hp.minkowski(p16)
    hat5, hat6 = hp.onShell(g5, g6, P, hcase, side='left')
    hatp16 = hp.massive(*(t1.vector + hat6.vector))
    negp16 = hp.massive(*(-t1.vector - hat6.vector))
    d1 = (hp.ttg(t1, hatp16, hat6, hcase[2], ref=g5)
          @ tthgg(negp16, tbar2, h3, g4, hat5, hcase[:2])) / (s16 - t1.mass**2)

    P = [tbar2, g4, g5]
    p245 = tbar2.vector + g4.vector + g5.vector
    s245 = hp.minkowski(p245)
    hat5, hat6 = hp.onShell(g5, g6, P, hcase, side='right')
    hatp245 = hp.massive(*(tbar2.vector + g4.vector + hat5.vector))
    negp245 = hp.massive(*(-tbar2.vector - g4.vector - hat5.vector))
    d2 = (tthg(t1, hatp245, h3, hat6, hcase[2], g5)
          @ hp.ttgg(negp245, tbar2, g4, hat5, hcase[:2])) / (s245 - t1.mass**2)

    P = [g4, g5]
    p45 = g4.vector + g5.vector
    s45 = hp.minkowski(p45)
    hat5, hat6 = hp.onShell(g5, g6, P, hcase, side='right')
    hatp45 = hp.massless(*(g4.vector + hat5.vector))
    negp45 = hp.massless(*(-g4.vector - hat5.vector))
    d3a = (tthgg(t1, tbar2, h3, negp45, hat6, ['+'] + list(hcase[2]))
           @ hp.ggg(hatp45, g4, hat5, ['-'] + list(hcase[:2]))) / s45
    d3b = (tthgg(t1, tbar2, h3, negp45, hat6, ['-'] + list(hcase[2]))
           @ hp.ggg(hatp45, g4, hat5, ['+'] + list(hcase[:2]))) / s45
    
    amp = d1 + d2 + d3a + d3b
    return amp

def tthqqg(t1, tbar2, h3, q4, qbar5, g6, hcase):
    P = [t1, g6]
    p16 = t1.vector + g6.vector
    s16 = hp.minkowski(p16)
    hat5, hat6 = hp.onShell(qbar5, g6, P, hcase, side='left')
    hatp16 = hp.massive(*(t1.vector + hat6.vector))
    negp16 = hp.massive(*(-t1.vector - hat6.vector))
    d1 = (hp.ttg(t1, hatp16, hat6, hcase[2], ref=qbar5)
          @ tthqq(negp16, tbar2, h3, q4, hat5, hcase[:2])) / (s16 - t1.mass**2)

    P = [tbar2, q4, qbar5]
    p245 = tbar2.vector + q4.vector + qbar5.vector
    s245 = hp.minkowski(p245)
    hat5, hat6 = hp.onShell(qbar5, g6, P, hcase, side='right')
    hatp245 = hp.massive(*(tbar2.vector + q4.vector + hat5.vector))
    negp245 = hp.massive(*(-tbar2.vector - q4.vector - hat5.vector))
    d2 = (tthg(t1, hatp245, h3, hat6, hcase[2], qbar5)
          @ hp.ttqq(negp245, tbar2, q4, hat5, hcase[:2])) / (s245 - t1.mass**2)

    P = [q4, qbar5]
    p45 = q4.vector + qbar5.vector
    s45 = hp.minkowski(p45)
    hat5, hat6 = hp.onShell(qbar5, g6, P, hcase, side='right')
    hatp45 = hp.massless(*(q4.vector + hat5.vector))
    negp45 = hp.massless(*(-q4.vector - hat5.vector))
    d3a = (tthgg(t1, tbar2, h3, negp45, hat6, ['+'] + list(hcase[2]))
           @ hp.gqq(hatp45, q4, hat5, ['-'] + list(hcase[:2]))) / s45
    d3b = (tthgg(t1, tbar2, h3, negp45, hat6, ['-'] + list(hcase[2]))
           @ hp.gqq(hatp45, q4, hat5, ['+'] + list(hcase[:2]))) / s45

    amp = d1 + d2 + d3a + d3b
    return amp

def tthgggg(t1, tbar2, h3, g4, g5, g6, g7, hcase):
    P = [t1, g7]
    p17 = t1.vector + g7.vector
    s17 = hp.minkowski(p17)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='left')
    hatp17 = hp.massive(*(t1.vector + hat7.vector))
    negp17 = hp.massive(*(-t1.vector - hat7.vector))
    d1 = (hp.ttg(t1, hatp17, hat7, hcase[3], ref=g6)
          @ tthggg(negp17, tbar2, h3, g4, g5, hat6, hcase[:3])) / (s17 - t1.mass**2)

    P = [tbar2, g4, g5, g6]
    p2456 = tbar2.vector + g4.vector + g5.vector + g6.vector
    s2456 = hp.minkowski(p2456)
    hat6, hat7 = hp.onShell(g6 ,g7, P, hcase, side='right')
    hatp2456 = hp.massive(*(tbar2.vector + g4.vector + g5.vector + hat6.vector))
    negp2456 = hp.massive(*(-tbar2.vector - g4.vector - g5.vector - hat6.vector))
    d2 = (tthg(t1, hatp2456, h3, g7, hcase[3], g6)
          @ hp.ttggg(negp2456, tbar2, g4, g5, hat6, hcase[:3])) / (s2456 - t1.mass**2)

    P = [g4, g5, g6]
    p456 = g4.vector + g5.vector + g6.vector
    s456 = hp.minkowski(p456)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp456 = hp.massless(*(g4.vector + g5.vector + hat6.vector))
    negp456 = hp.massless(*(-g4.vector - g5.vector - hat6.vector))
    d3a = (tthgg(t1, tbar2, h3, negp456, hat7, ['+'] + list(hcase[3]))
           @ hp.gggg(hatp456, g4, g5, hat6, ['-'] + list(hcase[:3]))) / s456
    d3b = (tthgg(t1, tbar2, h3, negp456, hat7, ['-'] + list(hcase[3]))
           @ hp.gggg(hatp456, g4, g5, hat6, ['+'] + list(hcase[:3]))) / s456

    P = [g5, g6]
    p56 = g5.vector + g6.vector
    s56 = hp.minkowski(p56)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp56 = hp.massless(*(g5.vector + hat6.vector))
    negp56 = hp.massless(*(-g5.vector - hat6.vector))
    hcase4p7a = list(hcase[0]) + ['+'] + list(hcase[3])
    hcase4p7b = list(hcase[0]) + ['-'] + list(hcase[3])
    d4a = (tthggg(t1, tbar2, h3, g4, negp56, hat7, hcase4p7a)
           @ hp.ggg(hatp56, g5, g6, ['-'] + list(hcase[1:3]))) / s56
    d4b = (tthggg(t1, tbar2, h3, g4, negp56, hat7, hcase4p7b)
           @ hp.ggg(hatp56, g5, g6, ['+'] + list(hcase[1:3]))) / s56
    
    amp = d1 + d2 + d3a + d3b + d4a + d4b
    return amp

def tthqqgg(t1, tbar2, h3, q4, qbar5, g6, g7, hcase):
    P = [t1, g7]
    p17 = t1.vector + g7.vector
    s17 = hp.minkowski(p17)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='left')
    hatp17 = hp.massive(*(t1.vector + hat7.vector))
    negp17 = hp.massive(*(-t1.vector - hat7.vector))
    d1 = (hp.ttg(t1, hatp17, hat7, hcase[3], ref=g6)
          @ tthqqg(negp17, tbar2, h3, q4, qbar5, hat6, hcase[:3])) / (s17 - t1.mass**2)

    P = [tbar2, q4, qbar5, g6]
    p2456 = tbar2.vector + q4.vector + qbar5.vector + g6.vector
    s2456 = hp.minkowski(p2456)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp2456 = hp.massive(*(tbar2.vector + q4.vector + qbar5.vector + hat6.vector))
    negp2456 = hp.massive(*(-tbar2.vector - q4.vector - qbar5.vector - hat6.vector))
    d2 = (tthg(t1, hatp2456, h3, g7, hcase[3], g6)
          @ hp.ttqqg(negp2456, tbar2, q4, qbar5, hat6, hcase[:3])) / (s2456 - t1.mass**2)

    P = [q4, qbar5, g6]
    p456 = q4.vector + qbar5.vector + g6.vector
    s456 = hp.minkowski(p456)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp456 = hp.massless(*(q4.vector + qbar5.vector + hat6.vector))
    negp456 = hp.massless(*(-q4.vector - qbar5.vector - hat6.vector))
    d3a = (tthgg(t1, tbar2, h3, negp456, hat7, ['+'] + list(hcase[3]))
           @ hp.gqqg(hatp456, q4, qbar5, hat6, ['-'] + list(hcase[:3]))) / s456
    d3b = (tthgg(t1, tbar2, h3, negp456, hat7, ['-'] + list(hcase[3]))
           @ hp.gqqg(hatp456, q4, qbar5, hat6, ['+'] + list(hcase[:3]))) / s456

    P = [qbar5, g6]
    p56 = qbar5.vector + g6.vector
    s56 = hp.minkowski(p56)
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp56 = hp.massless(*(qbar5.vector + hat6.vector))
    negp56 = hp.massless(*(-qbar5.vector - hat6.vector))
    hcase4p7a = list(hcase[0]) + ['+'] + list(hcase[3])
    hcase4p7b = list(hcase[0]) + ['-'] + list(hcase[3])
    d4a = (tthggg(t1, tbar2, h3, q4, negp56, hat7, hcase4p7a)
           @ hp.ggg(hatp56, qbar5, g6, ['-'] + list(hcase[1:3]))) / s56
    d4b = (tthggg(t1, tbar2, h3, q4, negp56, hat7, hcase4p7b)
           @ hp.ggg(hatp56, qbar5, g6, ['+'] + list(hcase[1:3]))) / s56
    amp = d1 + d2 + d3a + d3b + d4a + d4b
    return amp

def tthqqqq(t1, tbar2, h3, q4, qbar5, q6, qbar7, hcase):
    P = [t1, qbar7]
    p17 = t1.vector + qbar7.vector
    s17 = hp.minkowski(p17)
    hat6, hat7 = hp.onShell(q6, qbar7, P, hcase, side='left')
    hatp17 = hp.massive(*(t1.vector + hat7.vector))
    negp17 = hp.massive(*(-t1.vector - hat7.vector))
    d1 = (hp.ttg(t1, hatp17, hat7, hcase[3], ref=q6)
          @ tthqqg(negp17, tbar2, h3, q4, qbar5, hat6, hcase[:3])) / (s17 - t1.mass**2)

    P = [tbar2, q4, qbar5, q6]
    p2456 = tbar2.vector + q4.vector + qbar5.vector + q6.vector
    s2456 = hp.minkowski(p2456)
    hat6, hat7 = hp.onShell(q6, qbar7, P, hcase, side='right')
    hatp2456 = hp.massive(*(tbar2.vector + q4.vector + qbar5.vector + hat6.vector))
    negp2456 = hp.massive(*(-tbar2.vector - q4.vector - qbar5.vector - hat6.vector))
    d2 = (tthg(t1, hatp2456, h3, qbar7, hcase[3], q6)
          @ hp.ttqqg(negp2456, tbar2, q4, qbar5, hat6, hcase[:3])) / (s2456 - t1.mass**2)

    P = [q4, qbar5, q6]
    p456 = q4.vector + qbar5.vector + q6.vector
    s456 = hp.minkowski(p456)
    hat6, hat7 = hp.onShell(q6, qbar7, P, hcase, side='right')
    hatp456 = hp.massless(*(q4.vector + qbar5.vector + hat6.vector))
    negp456 = hp.massless(*(-q4.vector - qbar5.vector - hat6.vector))
    d3a = (tthgg(t1, tbar2, h3, negp456, hat7, ['+'] + list(hcase[3]))
           @ hp.gqqg(hatp456, q4, qbar5, hat6, ['-'] + list(hcase[:3]))) / s456
    d3b = (tthgg(t1, tbar2, h3, negp456, hat7, ['-'] + list(hcase[3]))
           @ hp.gqqg(hatp456, q4, qbar5, hat6, ['+'] + list(hcase[:3]))) / s456

    P = [qbar5, q6]
    p56 = qbar5.vector + q6.vector
    s56 = hp.minkowski(p56)
    hat6, hat7 = hp.onShell(q6, qbar7, P, hcase, side='right')
    hatp56 = hp.massless(*(qbar5.vector + hat6.vector))
    negp56 = hp.massless(*(-qbar5.vector - hat6.vector))
    hcase4p7a = list(hcase[0]) + ['+'] + list(hcase[3])
    hcase4p7b = list(hcase[0]) + ['-'] + list(hcase[3])
    d4a = (tthggg(t1, tbar2, h3, q4, negp56, hat7, hcase4p7a)
           @ hp.ggg(hatp56, qbar5, q6, ['-'] + list(hcase[1:3]))) / s56
    d4b = (tthggg(t1, tbar2, h3, q4, negp56, hat7, hcase4p7b)
           @ hp.ggg(hatp56, qbar5, q6, ['+'] + list(hcase[1:3]))) / s56
    amp = d1 + d2 + d3a + d3b + d4a + d4b
    return amp