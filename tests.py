import numpy as np
import amplitudes.helpers as hp
import amplitudes.core as core
import amplitudes.amps as amps

def ttqqg_test(momenta):
    t1 = hp.massive(momenta[0])
    tbar2 = hp.massive(momenta[1])
    q3 = hp.massless(momenta[2])
    qbar4 = hp.massless(momenta[3])
    g5 = hp.massless(momenta[4])
    hcase = ['-', '+', '+']
    ttqqg = core.ttqqg(t1, tbar2, q3, qbar4, g5, hcase)
    
    d1 = 1j * (((t1.sbra @ (qbar4.sketabra() + g5.sketabra() ) @ t1.momentum() @ g5.sket
    + hp.sbraket(t1, g5) @ hp.abraket(qbar4, g5) @ hp.sbraket(g5, qbar4)) @ hp.abraket(q3, tbar2)
    + hp.abraket(t1, q3) @ g5.sbra @ t1.sketabra() @ (qbar4.momentum() + g5.momentum()) @ tbar2.sket)
    * (qbar4.abra @ t1.momentum() @ g5.sket) / ((g5.abra @ t1.momentum() @ g5.sket)
                                            * hp.abraket(qbar4, g5) * hp.abraket(q3, qbar4) *
                                            (q3.sbra @ (qbar4.sketabra() + g5.sketabra()) @ t1.momentum() @ g5.sket)))

    p345 = q3.vector + qbar4.vector + g5.vector
    s345 = hp.minkowski(p345)
    d2 = 1j * ((t1.mass * hp.sbraket(q3, g5) * hp.sbraket(qbar4, g5)**2 * hp.abraket(t1, tbar2)) 
            / (s345 * hp.sbraket(q3, qbar4) * (q3.sbra @ (qbar4.sketabra() + g5.sketabra()) @ t1.momentum() @ g5.sket)))
    
    diff = ttqqg - (d1 + d2)
    print('\nTEST FOR ttqqg:')
    print('----------------------------------------------')
    if diff.all() <= 1:
        print('test for ttqqg passed!\n')
        print(f'amplitude from my ttqqg function:\n{ttqqg}\n')
        print(f'amplitude from explicit expression in Campbell2023:\n{d1 + d2}\n')
        print(f'|diff| =\n{abs(diff)}')
        print('----------------------------------------------\n')
    else:
        print('----------------------------------------------')
        print('test for ttqqg failed!')
        print('----------------------------------------------\n')
    return

def ttgg_test(phitop, thetatop, ptop, phiglu, thetaglu):
    mtop = 171
    Etop = np.sqrt(ptop**2 + mtop**2)
    ptopx = ptop * np.sin(thetatop) * np.cos(phitop)
    ptopy = ptop * np.sin(thetatop) * np.sin(phitop)
    ptopz = ptop * np.cos(thetatop)

    Eglu = Etop
    pglux = Eglu * np.sin(thetaglu) * np.cos(phiglu)
    pgluy = Eglu * np.sin(thetaglu) * np.sin(phiglu)
    pgluz = Eglu * np.cos(thetaglu)

    t1    = hp.massive([Etop, ptopx, ptopy, ptopz])
    tbar2 = hp.massive([Etop, -ptopx, -ptopy, -ptopz])
    print(hp.abraket(t1, tbar2))
    g3    = hp.massless([-Eglu, pglux, pgluy, pgluz])
    g4    = hp.massless([-Eglu, -pglux, -pgluy, -pgluz])
    hcases = [['+', '+'],
              ['-', '-'],
              ['+', '-'],
              ['-', '+']]

    hcases2 = [[hcase[1], hcase[0]] for hcase in hcases]

    amps1234 = [core.ttgg(t1, tbar2, g3, g4, hcase) for hcase in hcases]
    print(amps1234[0])
    amps1243 = [core.ttgg(t1, tbar2, g4, g3, hcase) for hcase in hcases2]
    subleadings = [amp1234 + amp1243 for amp1234, amp1243 in zip(amps1234, amps1243)]

    N = 3
    V = N**2 - 1
    colorSum = []
    for A1, A2, A3 in zip(amps1234, amps1243, subleadings):
         colorSum += [V * (N * (abs(A1**2) + abs(A2**2)) - (1 / N) * abs(A3**2))]

    spinSum = sum([np.sum(i) for i in colorSum])

    p12 = -t1.vector - tbar2.vector
    p13 = -t1.vector - g3.vector
    p14 = -t1.vector - g4.vector

    s = hp.minkowski(p12)
    t = hp.minkowski(p13)
    u = hp.minkowski(p14)

    Mss = (4 / s**2) * (t - mtop**2) * (u - mtop**2)
    Mtt = (-2 / (t - mtop**2)**2) * (4 * mtop**4 - (t - mtop**2) * (u - mtop**2) + 2 * mtop**2 * (t - mtop**2))
    Muu = (-2 / (u - mtop**2)**2) * (4 * mtop**4 - (u - mtop**2) * (t - mtop**2) + 2 * mtop**2 * (u - mtop**2))
    Mst = (4 / (s * (t - mtop**2))) * (mtop**4 - t * (s + t))
    Msu = (4 / (s * (u - mtop**2))) * (mtop**4 - u * (s + u))
    Mtu = ((-4 * mtop**2) / ((t - mtop**2) * (u - mtop**2))) * (4 * mtop**2 + (t - mtop**2) + (u - mtop**2))

    result = (12 * Mss + (16 / 3) * Mtt + (16 / 3) * Muu + 6 * Mst + 6 * Msu - (2 / 3) * Mtu) * 4

    diff = spinSum - result
    print('\nTEST FOR ttgg:')
    print('----------------------------------------------')
    if diff <= 1:
        print('test for ttgg passed!\n')
        print(f'cross section from my ttgg function:\n{spinSum}\n')
        print(f'cross-section from explicit expression in Gluck1978:\n{result}\n')
        print(f'diff =\n{diff}')
        print('----------------------------------------------\n')
    else:
        print('----------------------------------------------')
        print('test for ttgg failed!')
        print('----------------------------------------------\n')
    return


def tthg_test(phitop, thetatop, ptop, phiglu, thetaglu, hcase):
    mtop = 171
    mhig = 125
    Etop = np.sqrt(ptop**2 + mtop**2)
    ptopx = ptop * np.sin(thetatop) * np.cos(phitop)
    ptopy = ptop * np.sin(thetatop) * np.sin(phitop)
    ptopz = ptop * np.cos(thetatop)

    Eglu = Etop - (mhig**2 / (4 * Etop))
    pglux = Eglu * np.sin(thetaglu) * np.cos(phiglu)
    pgluy = Eglu * np.sin(thetaglu) * np.sin(phiglu)
    pgluz = Eglu * np.cos(thetaglu)

    Ehig = np.sqrt(Eglu**2 + mhig**2)
    t1    = hp.massive([Etop, ptopx, ptopy, ptopz])
    tbar2 = hp.massive([Etop, -ptopx, -ptopy, -ptopz])
    h3    = hp.massive([-Ehig, pglux, pgluy, pgluz])
    g4    = hp.massless([-Eglu, -pglux, -pgluy, -pgluz])
    ref1  = hp.massless([1, 1, 0, 0])
    ref2  = hp.massless([1, 0, 1, 0])
    ref3  = hp.massless([1, 0, 0, 1])
    ref4  = hp.massless([5, 3, -4, 0])

    tthg1 = core.tthg(t1, tbar2, h3, g4, hcase, ref1)
    tthg2 = core.tthg(t1, tbar2, h3, g4, hcase, ref2)
    tthg3 = core.tthg(t1, tbar2, h3, g4, hcase, ref3)
    tthg4 = core.tthg(t1, tbar2, h3, g4, hcase, ref4)

    print('\nTEST FOR tthg:')
    print('----------------------------------------------')
    if abs(tthg1 - tthg2).all() <= 1:
        print('test for tthg passed!\n')
        print(f'amplitude for ref1:\n{tthg1}\n')
        print(f'amplitude for ref2:\n{tthg2}\n')
        print(f'amplitude for ref3:\n{tthg3}\n')
        print(f'amplitude for ref4:\n{tthg4}\n')
        print('----------------------------------------------\n')
    else:
        print('----------------------------------------------')
        print('test for tthg failed!')
        print('----------------------------------------------\n')
    return