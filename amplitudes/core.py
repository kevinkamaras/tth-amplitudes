import helpers as hp

def tthg(t1, tbar2, h3, g4, hcase):
    amp = None
    return amp

def tthgg(t1, tbar2, h3, g4, g5, hcase):
    shift = hp.BCFW(g4, g5, hcase)
    
    p15 = t1 + g5
    hat4, hat5 = hp.onShell(shift, p15)
    hatp15 = t1 + hat5
    d1 = hp.ttg(t1, hatp15, hat5, hcase) * tthg(-hatp15, tbar2, h3, hat4, hcase) / (abs(p15)**2 - hp.mtop**2)

    p24 = tbar2 + g4
    hat4, hat5 = hp.onShell(shift, p24)
    hatp24 = tbar2 + hat4
    d2 = tthg(t1, -hatp24, h3, hat5, hcase) * hp.ttg(hatp24, tbar2, hat4, hcase) / (abs(p24)**2 - hp.mtop**2)
    
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
    d1 = hp.ttg(t1, hatp16, hat6) * tthgg(-hatp16, tbar2, h3, g4, hat5) / (abs(p16)**2 - hp.mtop**2)

    p136 = t1 + h3 + g6
    hat5, hat6 = hp.onShell(shift, p136)
    hatp136 = t1 + h3 + hat6
    d2 = tthg(t1, hatp136, h3, hat6) * hp.ttgg(-hatp136, tbar2, g4, hat5) / (abs(p136)**2 - hp.mtop**2)

    p45 = g4 + g5
    hat5, hat6 = hp.onShell(shift, p45)
    hatp45 = g4 + hat5
    d3 = tthgg(t1, tbar2, h3, -hatp45, hat6) * hp.ggg(hatp45, g4, hat5) / abs(p45)**2

    amp = d1 + d2 + d3
    return amp

def tthqqg(t1, tbar2, h3, q4, qbar5, g6, hcase):
    shift = hp.BCFW(qbar5, g6, hcase)

    p16 = t1 + g6
    hat5, hat6 = hp.onShell(shift, p16)
    hatp16 = t1 + hat6
    d1 = hp.ttg(t1, hatp16, hat6) * tthqq(-hatp16, tbar2, h3, q4, hat5) / (abs(p16)**2 - hp.mtop**2)

    p136 = t1 + h3 + g6
    hat5, hat6 = hp.onShell(shift, p136)
    hatp136 = t1 + h3 + hat6
    d2 = tthg(t1, hatp136, h3, hat6) * hp.ttqq(-hatp136, tbar2, q4, hat5) / (abs(p136)**2 - hp.mtop**2)

    p45 = q4 + qbar5
    hat5, hat6 = hp.onShell(shift, p45)
    hatp45 = q4 + hat5
    d3 = tthgg(t1, tbar2, h3, -hatp45, hat6) * hp.qqg(hatp45, q4, hat5) / abs(p45)**2

    amp = d1 + d2 + d3
    return amp

def tthgggg(t1, tbar2, h3, g4, g5, g6, g7, hcase):
    shift = hp.BCFW(g6, g7, hcase)

    p17 = t1 + g7
    hat6, hat7 = hp.onShell(shift, p17)
    hatp17 = t1 + hat7
    d1 = hp.ttg(t1, hatp17, hat7) * hp.tthggg(-hatp17, tbar2, h3, g4, g5, hat6) / (abs(p17)**2 - hp.mtop**2)

    p137 = t1 + h3 + g7
    hat6, hat7 = hp.onShell(shift, p137)
    hatp137 = t1 + h3 + hat7
    d2 = tthg(t1, hatp137, h3, g7) * hp.ttggg(-hatp137, tbar2, g4, g5, hat6) / (abs(p137)**2 - hp.mtop**2)

    p456 = g4 + g5 + g6
    hat6, hat7 = hp.onShell(shift, p456)
    hatp456 = g4 + g5 + hat6
    d3 = tthgg(t1, tbar2, h3, -hatp456, hat7) * hp.gggg(hatp456, g4, g5, hat6) / abs(p456)**2

    p56 = g5 + g6
    hat6, hat7 = hp.onShell(shift, p56)
    hatp56 = g5 + hat6
    d4 = tthggg(t1, tbar2, h3, g4, -hatp56, hat7) * hp.ggg(hatp56, g5, g6) / abs(p56)**2

    amp = d1 + d2 + d3 + d4
    return amp

def tthqqgg(t1, tbar2, h3, q4, qbar5, g6, g7, hcase):
    shift = hp.BCFW(g6, g7, hcase)

    p17 = t1 + g7
    hat6, hat7 = hp.onShell(shift, p17)
    hatp17 = t1 + hat7
    d1 = hp.ttg(t1, hatp17, hat7) * hp.tthqqg(-hatp17, tbar2, h3, q4, qbar5, hat6) / (abs(p17)**2 - hp.mtop**2)

    p137 = t1 + h3 + g7
    hat6, hat7 = hp.onShell(shift, p137)
    hatp137 = t1 + h3 + hat7
    d2 = tthg(t1, hatp137, h3, g7) * hp.ttqqg(-hatp137, tbar2, q4, qbar5, hat6) / (abs(p137)**2 - hp.mtop**2)

    p456 = q4 + qbar5 + g6
    hat6, hat7 = hp.onShell(shift, p456)
    hatp456 = q4 + qbar5 + hat6
    d3 = tthgg(t1, tbar2, h3, -hatp456, hat7) * hp.qqgg(hatp456, q4, qbar5, hat6) / abs(p456)**2

    p56 = qbar5 + g6
    hat6, hat7 = hp.onShell(shift, p56)
    hatp56 = qbar5 + hat6
    d4 = tthggg(t1, tbar2, h3, q4, -hatp56, hat7) * hp.ggg(hatp56, qbar5, g6) / abs(p56)**2

    amp = d1 + d2 + d3 + d4
    return amp

def tthqqqq(t1, tbar2, h3, q4, qbar5, q6, qbar7, hcase):
    amp = None
    return amp