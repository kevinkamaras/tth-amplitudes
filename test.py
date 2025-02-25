import numpy as np
import amplitudes.helpers as hp
import amplitudes.core as core

# g1 = hp.massless(*np.array([np.sqrt(2), 1, 0, 1]))
# bispinor = g1.aket @ (g1.sbra - 1j * g1.sbra)
# g2 = hp.massless(*np.array([np.trace(bispinor @ sigma) /2 for sigma in hp.pauli]))
# g3 = hp.massless(*(-g1.vector - g2.vector))

# t1    = hp.massive(*np.array([np.sqrt(2)+5j, 1-1j, 0, 1-1j]))
# tbar2 = hp.massive(*np.array([-2*np.sqrt(2)-5j, -2+1j, 0, -2+1j]))

# hcase = ['+']
# ref   = hp.massless(*np.array([np.sqrt(2), 1, -1, 0]))
ref1 = hp.massless(*[np.sqrt(5), 2, 1, 0])
ref2 = hp.massless(*[np.sqrt(14), 3, -1, 2])

# ttg1 = hp.ttg(t1, tbar2, g1, hcase, ref1)
# ttg2 = hp.ttg(t1, tbar2, g1, hcase, ref2)
# # print(ttg1)
# print(ttg2)

hcase1 = ['-', '-', '+']
hcase2 = ['+', '-', '-']
# ggg1 = hp.ggg(g1, g2, g3, hcase1)
# ggg2 = hp.ggg(g3, g1, g2, hcase2)

t1 = hp.massive(*[5, 1, 1, 1])
tbar2 = hp.massive(*[5, 1, -1, -1])
g4 = hp.massless(*[5, 4, 0, 3])
g5 = hp.massless(*[5, -3, 4, 0])
h3 = hp.massive(*(-t1.vector - tbar2.vector - g4.vector - g5.vector))
hcase1 = ['+', '+']
hcase2 = ['-', '+']
hcase3 = ['+', '-']
hcase4 = ['-', '-']

# tthgg1 = core.tthgg(t1, tbar2, h3, g4, g5, hcase1)
# tthgg2 = core.tthgg(t1, tbar2, h3, g4, g5, hcase2)
# tthgg3 = core.tthgg(t1, tbar2, h3, g4, g5, hcase3)
# tthgg4 = core.tthgg(t1, tbar2, h3, g4, g5, hcase4)
# print(tthgg1)
# print(tthgg2)
# print(tthgg3)
# print(tthgg4)

t1    = hp.massive(*[5, -3, -1, 0])
tbar2 = hp.massive(*[5, 0, 1, -4])
q3    = hp.massless(*[5, 3, 0, 4])
qbar4 = hp.massless(*[-7.5, 7.5, 0, 0])
g5    = hp.massless(*[-7.5, -7.5, 0, 0])
hcases = [['-', '+', '+'],
          ['+', '-', '-'],
          ['+', '-', '+'],
          ['-', '+', '-'],
          ['+', '+', '-'],
          ['-', '-', '+'],
          ['+', '+', '+'],
          ['-', '-', '-']]

# ttqqgs = [core.ttqqg(t1, tbar2, q3, qbar4, g5, hcase) for hcase in hcases]

d1 = (((t1.sbra @ (qbar4.sketabra + g5.sketabra) @ t1.momentum @ g5.sket
      + hp.sbraket(t1, g5) @ hp.abraket(qbar4, g5) @ hp.sbraket(g5, qbar4)) @ hp.abraket(q3, tbar2)
      + hp.abraket(t1, q3) @ g5.sbra @ t1.sketabra @ (qbar4.momentum + g5.momentum) @ tbar2.sket)
      * (qbar4.abra @ t1.momentum @ g5.sket) / ((g5.abra @ t1.momentum @ g5.sket)
                                                * hp.abraket(qbar4, g5) * hp.abraket(q3, qbar4)*
                                                (q3.sbra @ (qbar4.sketabra + g5.sketabra) @ t1.momentum @ g5.sket)))
# print(d1)
p345 = q3.vector + qbar4.vector + g5.vector
s345 = hp.minkowski(p345)
d2 = (t1.mass * hp.sbraket(q3, g5) * hp.sbraket(qbar4, g5)**2 * hp.abraket(t1, tbar2)
      / (s345 * hp.sbraket(q3, qbar4) * (q3.sbra @ (qbar4.sketabra + g5.sketabra) @ t1.momentum @ g5.sket)))
# print(d2)
# print(ttqqg)

m = 171

phitop = 1.27
thetatop = 0.78
ptop = 50
phiglu = 0.97
thetaglu = 2.5

Etop = np.sqrt(ptop**2 + m**2)
ptopx = ptop * np.sin(thetatop) * np.cos(phitop)
ptopy = ptop * np.sin(thetatop) * np.sin(phitop)
ptopz = ptop * np.cos(thetatop)

Eglu = Etop
pglux = Eglu * np.sin(thetaglu) * np.cos(phiglu)
pgluy = Eglu * np.sin(thetaglu) * np.sin(phiglu)
pgluz = Eglu * np.cos(thetaglu)

t1    = hp.massive(*[Etop, ptopx, ptopy, ptopz])
tbar2 = hp.massive(*[Etop, -ptopx, -ptopy, -ptopz])
g3    = hp.massless(*[-Eglu, pglux, pgluy, pgluz])
g4    = hp.massless(*[-Eglu, -pglux, -pgluy, -pgluz])
hcases = [['+', '+'],
          ['-', '-'],
          ['+', '-'],
          ['-', '+']]

ttggs = [hp.ttgg(t1, tbar2, g3, g4, hcase) for hcase in hcases]
for ttgg in ttggs:
    print(ttgg)