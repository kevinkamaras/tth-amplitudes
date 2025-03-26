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
    if (abs(diff) <= 1e-5).all():
        print('test for ttqqg passed!\n')
        print(f'amplitude from my ttqqg function:\n{ttqqg}\n')
        print(f'amplitude from explicit expression in Campbell2023:\n{d1 + d2}\n')
        print('the two amplitudes are equal')
        print('----------------------------------------------\n')
    else:
        print('----------------------------------------------')
        print('test for ttqqg failed!')
        print(f'amplitude from my ttqqg function:\n{ttqqg}\n')
        print(f'amplitude from explicit expression in Campbell2023:\n{d1 + d2}\n')
        print('----------------------------------------------\n')
    return

def ttgg_test(phitop, thetatop, ptop, phiglu, thetaglu):
    v = 0.9
    phi = 1.5
    theta = 0.6

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
    
    mtop = 171
    Etop = np.sqrt(ptop**2 + mtop**2)
    ptopx = ptop * np.sin(thetatop) * np.cos(phitop)
    ptopy = ptop * np.sin(thetatop) * np.sin(phitop)
    ptopz = ptop * np.cos(thetatop)

    Eglu = Etop
    pglux = Eglu * np.sin(thetaglu) * np.cos(phiglu)
    pgluy = Eglu * np.sin(thetaglu) * np.sin(phiglu)
    pgluz = Eglu * np.cos(thetaglu)

    t1    = [Etop, ptopx, ptopy, ptopz]
    tbar2 = [Etop, -ptopx, -ptopy, -ptopz]
    g3    = [-Eglu, pglux, pgluy, pgluz]
    g4    = [-Eglu, -pglux, -pgluy, -pgluz]

    t1_b    = L @ t1
    tbar2_b = L @ tbar2
    g3_b    = L @ g3
    g4_b    = L @ g4

    t1 = hp.massive(t1_b)
    tbar2 = hp.massive(tbar2_b)
    g3 = hp.massless(g3_b)
    g4 = hp.massless(g4_b)

    hcases = [['+', '+'],
              ['-', '-'],
              ['+', '-'],
              ['-', '+']]

    hcases2 = [[hcase[1], hcase[0]] for hcase in hcases]

    amps1234 = [core.ttgg(t1, tbar2, g3, g4, hcase) for hcase in hcases]
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
        # print(f'diff =\n{diff}')
        print('the cross sections are equal')
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
    if (np.sum(abs(tthg1 - tthg2)) <= 1e-5 and
        np.sum(abs(tthg1 - tthg3)) <= 1e-5 and
        np.sum(abs(tthg1 - tthg4)) <= 1e-5):
        print('test for tthg passed!\n')
        print(f'amplitude for ref1:\n{tthg1}\n')
        print(f'amplitude for ref2:\n{tthg2}\n')
        print(f'amplitude for ref3:\n{tthg3}\n')
        print(f'amplitude for ref4:\n{tthg4}\n')
        print('the amplitude is independent of the reference spinor')
        print('----------------------------------------------\n')
    else:
        print('----------------------------------------------')
        print('test for tthg failed!')
        print('----------------------------------------------\n')
    return

def boost_ttgg(v, phi, theta):
    mtop = 171
    phitop = 1.27
    thetatop = 0.78
    ptop = 200
    phiglu = 0.97
    thetaglu = 2.5
    Etop = np.sqrt(ptop**2 + mtop**2)
    ptopx = ptop * np.sin(thetatop) * np.cos(phitop)
    ptopy = ptop * np.sin(thetatop) * np.sin(phitop)
    ptopz = ptop * np.cos(thetatop)

    Eglu = Etop
    pglux = Eglu * np.sin(thetaglu) * np.cos(phiglu)
    pgluy = Eglu * np.sin(thetaglu) * np.sin(phiglu)
    pgluz = Eglu * np.cos(thetaglu)

    t1    = np.array([Etop, ptopx, ptopy, ptopz])
    tbar2 = np.array([Etop, -ptopx, -ptopy, -ptopz])
    g3    = np.array([-Eglu, pglux, pgluy, pgluz])
    g4    = np.array([-Eglu, -pglux, -pgluy, -pgluz])
    momenta = [t1, tbar2, g3, g4]
    momenta_b = hp.boost(momenta, v, phi, theta)
    hcase = ['+', '-']
    
    t1    = hp.massive(t1)
    tbar2 = hp.massive(tbar2)
    g3    = hp.massless(g3)
    g4    = hp.massless(g4)

    t1_b    = hp.massive(momenta_b[0])
    tbar2_b = hp.massive(momenta_b[1])
    g3_b    = hp.massless(momenta_b[2])
    g4_b    = hp.massless(momenta_b[3])

    ttgg = core.ttgg(t1, tbar2, g3, g4, hcase)
    ttgg_b = core.ttgg(t1_b, tbar2_b, g3_b, g4_b, hcase)

    spinSum = np.sum(abs(ttgg)**2)
    spinSum_b = np.sum(abs(ttgg_b)**2)

    diff = abs(spinSum) - abs(spinSum_b)

    print('\nTEST FOR ttgg BOOST:')
    print('----------------------------------------------')
    if (abs(diff) <= 1e-5).all():
        print('test for ttgg boost passed!\n')
        print(f'unboosted spin sum:\n{spinSum}\n')
        print(f'boosted spin sum:\n{spinSum}\n')
        print(f'|diff| =\n{abs(diff)}')
        print('the two amplitudes are equal')
        print('----------------------------------------------\n')
    else:
        print('test for ttgg boost failed!\n')
        print(f'unboosted spin sum:\n{spinSum}\n')
        print(f'boosted spin sum:\n{spinSum_b}\n')
        print('----------------------------------------------\n')
    return

def boost_tthg(v, phi, theta):
    mtop = 171
    mhig = 125

    phitop = 1.27
    thetatop = 0.78
    ptop = 200
    phiglu = 0.97
    thetaglu = 2.5
    hcases = ['+', '-']

    Etop = np.sqrt(ptop**2 + mtop**2)
    ptopx = ptop * np.sin(thetatop) * np.cos(phitop)
    ptopy = ptop * np.sin(thetatop) * np.sin(phitop)
    ptopz = ptop * np.cos(thetatop)

    Eglu = Etop - (mhig**2 / (4 * Etop))
    pglux = Eglu * np.sin(thetaglu) * np.cos(phiglu)
    pgluy = Eglu * np.sin(thetaglu) * np.sin(phiglu)
    pgluz = Eglu * np.cos(thetaglu)

    Ehig = np.sqrt(Eglu**2 + mhig**2)
    t1    = np.array([Etop, ptopx, ptopy, ptopz])
    tbar2 = np.array([Etop, -ptopx, -ptopy, -ptopz])
    h3    = np.array([-Ehig, pglux, pgluy, pgluz])
    g4    = np.array([-Eglu, -pglux, -pgluy, -pgluz])
    momenta = [t1, tbar2, h3, g4]
    momenta_b = hp.boost(momenta, v, phi, theta)
    ref  = [1, 1, 0, 0]

    tthgs = np.array([amps.tthg(*momenta, hcase, ref) for hcase in hcases])
    tthgs_b = np.array([amps.tthg(*momenta_b, hcase, ref) for hcase in hcases])

    spinSum = np.sum(abs(tthgs)**2)
    spinSum_b = np.sum(abs(tthgs_b)**2)

    diff = abs(spinSum) - abs(spinSum_b)

    print('\nTEST FOR tthg BOOST:')
    print('----------------------------------------------')
    if (abs(diff) <= 1e-5).all():
        print('test for tthg boost passed!\n')
        print(f'unboosted spin sum:\n{spinSum}\n')
        print(f'boosted spin sum:\n{spinSum}\n')
        print(f'|diff| =\n{abs(diff)}')
        print('the two spin sums are equal')
        print('----------------------------------------------\n')
    else:
        print('test for tthg boost failed!\n')
        print(f'unboosted amplitude:\n{spinSum}\n')
        print(f'boosted amplitude:\n{spinSum_b}\n')
        print('----------------------------------------------\n')
    return

def boost_tthgg(v, phi, theta):
    phi1   = 2.02
    theta1 = 0.4
    phi2   = 1.2
    theta2 = 0.9
    phi4   = 0.9
    theta4 = 1.2
    phi5   = 2.1
    theta5 = 0.45

    angles = [phi1, theta1, phi2, theta2, phi4, theta4, phi5, theta5]
    momenta = hp.tthggMomenta(angles)
    hcase = ['-', '+']

    momenta_b = hp.boost(momenta, v, phi, theta)

    tthgg = amps.tthgg(*momenta, hcase)
    tthgg_b = amps.tthgg(*momenta_b, hcase)

    diff = np.sum(abs(tthgg)**2) - np.sum(abs(tthgg_b)**2)

    print('\nTEST FOR tthgg BOOST:')
    print('----------------------------------------------')
    if abs(diff) <= 1e-8:
        print('test for tthgg boost passed!\n')
        print(f'unboosted spin sum:\n{np.sum(abs(tthgg)**2)}\n')
        print(f'boosted spin sum:\n{np.sum(abs(tthgg_b)**2)}\n')
        print(f'|diff| =\n{abs(diff)}')
        print('the two spin sums are equal')
        print('----------------------------------------------\n')
    else:
        print('test for tthgg boost failed!\n')
        print(f'unboosted spin sum:\n{np.sum(abs(tthgg)**2)}\n')
        print(f'boosted spin sum:\n{np.sum(abs(tthgg_b)**2)}\n')
        print(f'|diff| =\n{abs(diff)}')
        print('----------------------------------------------\n')
    return

def boost_tthgggg(v, phi, theta):
    mTop = 171
    mHiggs = 125

    phiTop = 1.02
    thetaTop = 0.87
    pTop = 200
    phiGlu1 = 1.2
    thetaGlu1 = 0.9
    phiGlu2 = 0.31
    thetaGlu2 = 1

    Etop = np.sqrt(mTop**2 + pTop**2)
    ptopx = pTop * np.sin(thetaTop) * np.cos(phiTop)
    ptopy = pTop * np.sin(thetaTop) * np.sin(phiTop)
    ptopz = pTop * np.cos(thetaTop)

    Eglu = (2 * Etop + mHiggs) / 4
    pglux1 = Eglu * np.sin(thetaGlu1) * np.cos(phiGlu1)
    pgluy1 = Eglu * np.sin(thetaGlu1) * np.sin(phiGlu1)
    pgluz1 = Eglu * np.cos(thetaGlu1)

    pglux2 = Eglu * np.sin(thetaGlu2) * np.cos(phiGlu2)
    pgluy2 = Eglu * np.sin(thetaGlu2) * np.sin(phiGlu2)
    pgluz2 = Eglu * np.cos(thetaGlu2)

    t1    = np.array([Etop, ptopx, ptopy, ptopz])
    tbar2 = np.array([Etop, -ptopx, -ptopy, -ptopz])
    h3    = np.array([mHiggs, 0, 0, 0])
    g4    = np.array([-Eglu, pglux1, pgluy1, pgluz1])
    g5    = np.array([-Eglu, pglux2, pgluy2, pgluz2])
    g6    = np.array([-Eglu, -pglux1, -pgluy1, -pgluz1])
    g7    = np.array([-Eglu, -pglux2, -pgluy2, -pgluz2])
    momenta = [t1, tbar2, h3, g4, g5, g6, g7]
    momenta_b = hp.boost(momenta, v, phi, theta)

    hcase = ['-', '+', '-', '+']
    tthgggg = amps.tthgggg(*momenta, hcase)
    tthgggg_b = amps.tthgggg(*momenta_b, hcase)

    diff = np.sum(abs(tthgggg)**2) - np.sum(abs(tthgggg_b)**2)

    print('\nTEST FOR tthgggg BOOST:')
    print('----------------------------------------------')
    if (abs(diff) <= 1).all():
        print('test for tthgggg boost passed!\n')
        print(f'unboosted spin sum:\n{tthgggg}\n')
        print(f'boosted spin sum:\n{tthgggg_b}\n')
        print(f'|diff| =\n{abs(diff)}')
        print('the two amplitudes are equal')
        print('----------------------------------------------\n')
    else:
        print('test for tthgggg boost failed!\n')
        print(f'unboosted spin sum:\n{np.sum(abs(tthgggg)**2)}\n')
        print(f'boosted spin sum:\n{np.sum(abs(tthgggg_b)**2)}\n')
        print('----------------------------------------------\n')
    return