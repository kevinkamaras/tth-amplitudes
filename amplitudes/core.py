import helpers as hp
import numpy as np

def tthg(t1, tbar2, h3, g4, hcase):
    amp = None
    return amp

def tthgg(t1, tbar2, h3, g4, g5, hcase):
    P = [t1, g5]
    p15 = np.sum([p.vector for p in P])
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='left')
    hatp15 = t1.vector + hat5.vector
    d1 = hp.ttg(t1, hatp15, hat5, hcase[1], ref=g4) * tthg(-hatp15, tbar2, h3, hat4, hcase[0]) / (np.vdot(p15, p15) - hp.mtop**2)

    P = [tbar2, g4]
    p24 = np.sum([p.vector for p in P])
    hat4, hat5 = hp.onShell(g4, g5, P, hcase, side='right')
    hatp24 = tbar2.vector + hat4.vector
    d2 = tthg(t1, -hatp24, h3, hat5, hcase[1]) * hp.ttg(hatp24, tbar2, hat4, hcase[0], ref=g5) / (np.vdot(p24, p24) - hp.mtop**2)
    
    amp = d1 + d2
    return amp

def tthqq(t1, tbar2, h3, q4, qbar5, hcase):
    amp = None
    return amp

def tthggg(t1, tbar2, h3, g4, g5, g6, hcase):
    P = [t1, g6]
    p16 = np.sum([p.vector for p in P])
    hat5, hat6 = hp.onShell(g5, g6, P, hcase, side='left')
    hatp16 = t1.vector + hat6.vector
    d1 = hp.ttg(t1, hatp16, hat6, hcase[2], ref=g5) * tthgg(-hatp16, tbar2, h3, g4, hat5, hcase[:2]) / (np.vdot(p16, p16) - hp.mtop**2)

    P = [tbar2, g4, g5]
    p245 = np.sum([p.vector for p in P])
    hat5, hat6 = hp.onShell(g5, g6, P, hcase, side='right')
    hatp245 = tbar2.vector + g4.vector + hat5.vector
    d2 = tthg(t1, hatp245, h3, hat6, hcase[2]) * hp.ttgg(-hatp245, tbar2, g4, hat5, hcase[:2]) / (np.vdot(p245, p245) - hp.mtop**2)

    P = [g4, g5]
    p45 = np.sum([p.vector for p in P])
    hat5, hat6 = hp.onShell(g5, g6, P, hcase, side='right')
    hatp45 = g4.vector + hat5.vector
    d3a = tthgg(t1, tbar2, h3, -hatp45, hat6, ['+'] + list(hcase[2])) * hp.ggg(hatp45, g4, hat5, ['-'] + list(hcase[:2])) / np.vdot(p45, p45)
    d3b = tthgg(t1, tbar2, h3, -hatp45, hat6, ['-'] + list(hcase[2])) * hp.ggg(hatp45, g4, hat5, ['+'] + list(hcase[:2])) / np.vdot(p45, p45)

    amp = d1 + d2 + d3a + d3b
    return amp

def tthqqg(t1, tbar2, h3, q4, qbar5, g6, hcase):
    P = [t1, g6]
    p16 = np.sum([p.vector for p in P])
    hat5, hat6 = hp.onShell(qbar5, g6, P, hcase, side='left')
    hatp16 = t1.vector + hat6.vector
    d1 = hp.ttg(t1, hatp16, hat6, hcase[2], ref=qbar5) * tthqq(-hatp16, tbar2, h3, q4, hat5, hcase[:2]) / (np.vdot(p16, p16) - hp.mtop**2)

    P = [tbar2, q4, qbar5]
    p245 = np.sum([p.vector for p in P])
    hat5, hat6 = hp.onShell(qbar5, g6, P, hcase, side='right')
    hatp245 = tbar2.vector + q4.vector + hat5.vector
    d2 = tthg(t1, hatp245, h3, hat6, hcase[2]) * hp.ttqq(-hatp245, tbar2, q4, hat5, hcase[:2]) / (np.vdot(p245, p245) - hp.mtop**2)

    P = [q4, qbar5]
    p45 = np.sum([p.vector for p in P])
    hat5, hat6 = hp.onShell(qbar5, g6, P, hcase, side='right')
    hatp45 = q4.vector + hat5.vector
    d3a = tthgg(t1, tbar2, h3, -hatp45, hat6, ['+'] + list(hcase[2])) * hp.qqg(hatp45, q4, hat5, ['-'] + list(hcase[:2])) / np.vdot(p45, p45)
    d3b = tthgg(t1, tbar2, h3, -hatp45, hat6, ['-'] + list(hcase[2])) * hp.qqg(hatp45, q4, hat5, ['+'] + list(hcase[:2])) / np.vdot(p45, p45)

    amp = d1 + d2 + d3a + d3b
    return amp

def tthgggg(t1, tbar2, h3, g4, g5, g6, g7, hcase):
    P = [t1, g7]
    p17 = np.sum([p.vector for p in P])
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='left')
    hatp17 = t1.vector + hat7.vector
    d1 = hp.ttg(t1, hatp17, hat7, hcase[3], ref=g6) * tthggg(-hatp17, tbar2, h3, g4, g5, hat6, hcase[:3]) / (np.vdot(p17, p17) - hp.mtop**2)

    p = [tbar2, g4, g5, g6]
    p2456 = np.sum([p.vector for p in P])
    hat6, hat7 = hp.onShell(g6 ,g7, P, hcase, side='right')
    hatp2456 = tbar2.vector + g4.vector + g5.vector + hat6.vector
    d2 = tthg(t1, hatp2456, h3, g7, hcase[3]) * hp.ttggg(-hatp2456, tbar2, g4, g5, hat6, hcase[:3]) / (np.vdot(p2456, p2456) - hp.mtop**2)

    P = [g4, g5, g6]
    p456 = np.sum([p.vector for p in P])
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp456 = g4.vector + g5.vector + hat6.vector
    d3a = tthgg(t1, tbar2, h3, -hatp456, hat7, ['+'] + list(hcase[3])) * hp.gggg(hatp456, g4, g5, hat6, ['-'] + list(hcase[:3])) / np.vdot(p456, p456)
    d3b = tthgg(t1, tbar2, h3, -hatp456, hat7, ['-'] + list(hcase[3])) * hp.gggg(hatp456, g4, g5, hat6, ['+'] + list(hcase[:3])) / np.vdot(p456, p456)

    P = [g5, g6]
    p56 = np.sum([p.vector for p in P])
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp56 = g5.vector + hat6.vector
    hcase4p7a = list(hcase[0]) + ['+'] + list(hcase[3])
    hcase4p7b = list(hcase[0]) + ['-'] + list(hcase[3])
    d4a = tthggg(t1, tbar2, h3, g4, -hatp56, hat7, hcase4p7a) * hp.ggg(hatp56, g5, g6, ['-'] + list(hcase[1:3])) / np.vdot(p56, p56)
    d4b = tthggg(t1, tbar2, h3, g4, -hatp56, hat7, hcase4p7b) * hp.ggg(hatp56, g5, g6, ['+'] + list(hcase[1:3])) / np.vdot(p56, p56)

    amp = d1 + d2 + d3a + d3b + d4a + d4b
    return amp

def tthqqgg(t1, tbar2, h3, q4, qbar5, g6, g7, hcase):
    P = [t1, g7]
    p17 = np.sum([p.vector for p in P])
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='left')
    hatp17 = t1.vector + hat7.vector
    d1 = hp.ttg(t1, hatp17, hat7, hcase[3], ref=g6) * tthqqg(-hatp17, tbar2, h3, q4, qbar5, hat6, hcase[:3]) / (np.vdot(p17, p17) - hp.mtop**2)

    P = [tbar2, q4, qbar5, g6]
    p2456 = np.sum([p.vector for p in P])
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp137 = tbar2.vector + q4.vector + qbar5.vector + hat6.vector
    d2 = tthg(t1, hatp137, h3, g7, hcase[3]) * hp.ttqqg(-hatp137, tbar2, q4, qbar5, hat6, hcase[:3]) / (np.vdot(p2456, p2456) - hp.mtop**2)

    P = [q4, qbar5, g6]
    p456 = np.sum([p.vector for p in P])
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp456 = q4.vector + qbar5.vector + hat6.vector
    d3a = tthgg(t1, tbar2, h3, -hatp456, hat7, ['+'] + list(hcase[3])) * hp.qqgg(hatp456, q4, qbar5, hat6, ['-'] + list(hcase[:3])) / np.vdot(p456, p456)
    d3b = tthgg(t1, tbar2, h3, -hatp456, hat7, ['-'] + list(hcase[3])) * hp.qqgg(hatp456, q4, qbar5, hat6, ['+'] + list(hcase[:3])) / np.vdot(p456, p456)

    P = [qbar5, g6]
    p56 = np.sum([p.vector for p in P])
    hat6, hat7 = hp.onShell(g6, g7, P, hcase, side='right')
    hatp56 = qbar5.vector + hat6.vector
    hcase4p7a = list(hcase[0]) + ['+'] + list(hcase[3])
    hcase4p7b = list(hcase[0]) + ['-'] + list(hcase[3])
    d4a = tthggg(t1, tbar2, h3, q4, -hatp56, hat7, hcase4p7a) * hp.ggg(hatp56, qbar5, g6, ['-'] + list(hcase[1:3])) / np.vdot(p56, p56)
    d4b = tthggg(t1, tbar2, h3, q4, -hatp56, hat7, hcase4p7b) * hp.ggg(hatp56, qbar5, g6, ['+'] + list(hcase[1:3])) / np.vdot(p56, p56)

    amp = d1 + d2 + d3a + d3b + d4a + d4b
    return amp

def tthqqqq(t1, tbar2, h3, q4, qbar5, q6, qbar7, hcase):
    amp = None
    return amp