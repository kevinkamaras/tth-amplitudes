import helpers as hp

def tthg(t1, tbar2, h3, g4, hcase):
    amp = None
    return amp

def tthgg(t1, tbar2, h3, g4, g5, hcase):
    shift = hp.BCFW(g4, g5, hcase)
    
    p15 = t1 + g5
    hat4, hat5 = hp.onShell(shift, p15)
    hatp15 = t1 + hat5
    d1 = hp.ttg(t1, hatp15, hat5, hcase[1], ref=g4) * tthg(-hatp15, tbar2, h3, hat4, hcase[0]) / (abs(p15)**2 - hp.mtop**2)

    p24 = tbar2 + g4
    hat4, hat5 = hp.onShell(shift, p24)
    hatp24 = tbar2 + hat4
    d2 = tthg(t1, -hatp24, h3, hat5, hcase[1]) * hp.ttg(hatp24, tbar2, hat4, hcase[0], ref=g5) / (abs(p24)**2 - hp.mtop**2)
    
    amp = d1 + d2
    return amp

def tthqq(t1, tbar2, h3, q4, qbar5, hcase):
    amp = None
    return amp

def tthggg(t1, tbar2, h3, g4, g5, g6, hcase):
    shift = hp.BCFW(g5, g6, hcase)

    p16 = t1 + g6
    hat5, hat6 = hp.onShell(shift, p16)
    hatp16 = t1 + hat6
    d1 = hp.ttg(t1, hatp16, hat6, hcase[2], ref=g5) * tthgg(-hatp16, tbar2, h3, g4, hat5, hcase[:2]) / (abs(p16)**2 - hp.mtop**2)

    p136 = t1 + h3 + g6
    hat5, hat6 = hp.onShell(shift, p136)
    hatp136 = t1 + h3 + hat6
    d2 = tthg(t1, hatp136, h3, hat6, hcase[2]) * hp.ttgg(-hatp136, tbar2, g4, hat5, hcase[:2]) / (abs(p136)**2 - hp.mtop**2)

    p45 = g4 + g5
    hat5, hat6 = hp.onShell(shift, p45)
    hatp45 = g4 + hat5
    d3a = tthgg(t1, tbar2, h3, -hatp45, hat6, ['+'] + list(hcase[2])) * hp.ggg(hatp45, g4, hat5, ['-'] + list(hcase[:2])) / abs(p45)**2
    d3b = tthgg(t1, tbar2, h3, -hatp45, hat6, ['-'] + list(hcase[2])) * hp.ggg(hatp45, g4, hat5, ['+'] + list(hcase[:2])) / abs(p45)**2

    amp = d1 + d2 + d3a + d3b
    return amp

def tthqqg(t1, tbar2, h3, q4, qbar5, g6, hcase):
    shift = hp.BCFW(qbar5, g6, hcase)

    p16 = t1 + g6
    hat5, hat6 = hp.onShell(shift, p16)
    hatp16 = t1 + hat6
    d1 = hp.ttg(t1, hatp16, hat6, hcase[2], ref=qbar5) * tthqq(-hatp16, tbar2, h3, q4, hat5, hcase[:2]) / (abs(p16)**2 - hp.mtop**2)

    p136 = t1 + h3 + g6
    hat5, hat6 = hp.onShell(shift, p136)
    hatp136 = t1 + h3 + hat6
    d2 = tthg(t1, hatp136, h3, hat6, hcase[2]) * hp.ttqq(-hatp136, tbar2, q4, hat5, hcase[:2]) / (abs(p136)**2 - hp.mtop**2)

    p45 = q4 + qbar5
    hat5, hat6 = hp.onShell(shift, p45)
    hatp45 = q4 + hat5
    d3a = tthgg(t1, tbar2, h3, -hatp45, hat6, ['+'] + list(hcase[2])) * hp.qqg(hatp45, q4, hat5, ['-'] + list(hcase[:2])) / abs(p45)**2
    d3b = tthgg(t1, tbar2, h3, -hatp45, hat6, ['-'] + list(hcase[2])) * hp.qqg(hatp45, q4, hat5, ['+'] + list(hcase[:2])) / abs(p45)**2

    amp = d1 + d2 + d3a + d3b
    return amp

def tthgggg(t1, tbar2, h3, g4, g5, g6, g7, hcase):
    shift = hp.BCFW(g6, g7, hcase)

    p17 = t1 + g7
    hat6, hat7 = hp.onShell(shift, p17)
    hatp17 = t1 + hat7
    d1 = hp.ttg(t1, hatp17, hat7, hcase[3], ref=g6) * hp.tthggg(-hatp17, tbar2, h3, g4, g5, hat6, hcase[:3]) / (abs(p17)**2 - hp.mtop**2)

    p137 = t1 + h3 + g7
    hat6, hat7 = hp.onShell(shift, p137)
    hatp137 = t1 + h3 + hat7
    d2 = tthg(t1, hatp137, h3, g7, hcase[3]) * hp.ttggg(-hatp137, tbar2, g4, g5, hat6, hcase[:3]) / (abs(p137)**2 - hp.mtop**2)

    p456 = g4 + g5 + g6
    hat6, hat7 = hp.onShell(shift, p456)
    hatp456 = g4 + g5 + hat6
    d3a = tthgg(t1, tbar2, h3, -hatp456, hat7, ['+'] + list(hcase[3])) * hp.gggg(hatp456, g4, g5, hat6, ['-'] + list(hcase[:3])) / abs(p456)**2
    d3b = tthgg(t1, tbar2, h3, -hatp456, hat7, ['-'] + list(hcase[3])) * hp.gggg(hatp456, g4, g5, hat6, ['+'] + list(hcase[:3])) / abs(p456)**2

    p56 = g5 + g6
    hat6, hat7 = hp.onShell(shift, p56)
    hatp56 = g5 + hat6
    hcase4p7a = list(hcase[0]) + ['+'] + list(hcase[3])
    hcase4p7b = list(hcase[0]) + ['-'] + list(hcase[3])
    d4a = tthggg(t1, tbar2, h3, g4, -hatp56, hat7, hcase4p7a) * hp.ggg(hatp56, g5, g6, ['-'] + list(hcase[1:3])) / abs(p56)**2
    d4b = tthggg(t1, tbar2, h3, g4, -hatp56, hat7, hcase4p7b) * hp.ggg(hatp56, g5, g6, ['+'] + list(hcase[1:3])) / abs(p56)**2

    amp = d1 + d2 + d3a + d3b + d4a + d4b
    return amp

def tthqqgg(t1, tbar2, h3, q4, qbar5, g6, g7, hcase):
    shift = hp.BCFW(g6, g7, hcase)

    p17 = t1 + g7
    hat6, hat7 = hp.onShell(shift, p17)
    hatp17 = t1 + hat7
    d1 = hp.ttg(t1, hatp17, hat7, hcase[3], ref=g6) * hp.tthqqg(-hatp17, tbar2, h3, q4, qbar5, hat6, hcase[:3]) / (abs(p17)**2 - hp.mtop**2)

    p137 = t1 + h3 + g7
    hat6, hat7 = hp.onShell(shift, p137)
    hatp137 = t1 + h3 + hat7
    d2 = tthg(t1, hatp137, h3, g7, hcase[3]) * hp.ttqqg(-hatp137, tbar2, q4, qbar5, hat6, hcase[:3]) / (abs(p137)**2 - hp.mtop**2)

    p456 = q4 + qbar5 + g6
    hat6, hat7 = hp.onShell(shift, p456)
    hatp456 = q4 + qbar5 + hat6
    d3a = tthgg(t1, tbar2, h3, -hatp456, hat7, ['+'] + list(hcase[3])) * hp.qqgg(hatp456, q4, qbar5, hat6, ['-'] + list(hcase[:3])) / abs(p456)**2
    d3b = tthgg(t1, tbar2, h3, -hatp456, hat7, ['-'] + list(hcase[3])) * hp.qqgg(hatp456, q4, qbar5, hat6, ['+'] + list(hcase[:3])) / abs(p456)**2

    p56 = qbar5 + g6
    hat6, hat7 = hp.onShell(shift, p56)
    hatp56 = qbar5 + hat6
    hcase4p7a = list(hcase[0]) + ['+'] + list(hcase[3])
    hcase4p7b = list(hcase[0]) + ['-'] + list(hcase[3])
    d4a = tthggg(t1, tbar2, h3, q4, -hatp56, hat7, hcase4p7a) * hp.ggg(hatp56, qbar5, g6, ['-'] + list(hcase[1:3])) / abs(p56)**2
    d4b = tthggg(t1, tbar2, h3, q4, -hatp56, hat7, hcase4p7b) * hp.ggg(hatp56, qbar5, g6, ['+'] + list(hcase[1:3])) / abs(p56)**2

    amp = d1 + d2 + d3a + d3b + d4a + d3b
    return amp

def tthqqqq(t1, tbar2, h3, q4, qbar5, q6, qbar7, hcase):
    amp = None
    return amp