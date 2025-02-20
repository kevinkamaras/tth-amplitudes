import numpy as np
import amplitudes.helpers as hp
import amplitudes.core as core

g1 = hp.massless(*np.array([np.sqrt(2), 1, 0, 1]))
bispinor = g1.aket @ (g1.sbra - 1j * g1.sbra)
g2 = hp.massless(*np.array([np.trace(bispinor @ sigma) /2 for sigma in hp.pauli]))
g3 = hp.massless(*(-g1.vector - g2.vector))

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
ggg1 = hp.ggg(g1, g2, g3, hcase1)
ggg2 = hp.ggg(g3, g1, g2, hcase2)

t1 = hp.massive(*[5, 1, 1, 1])
tbar2 = hp.massive(*[5, 1, -1, -1])
g4 = hp.massless(*[5, 4, 0, 3])
h3 = hp.massive(*(-t1.vector - tbar2.vector - g4.vector))
hcase = ['+']

tthg1 = core.tthg(t1, tbar2, h3, g4, hcase, ref1)
tthg2 = core.tthg(t1, tbar2, h3, g4, hcase, ref2)
print(tthg1)
print(tthg2)