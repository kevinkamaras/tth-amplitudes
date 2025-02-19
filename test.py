import numpy as np
import amplitudes.helpers as hp
import amplitudes.core as core

t1    = hp.massive(*np.array([5, 1, 1, 0]))
tbar2 = hp.massive(*np.array([5, 1, 0, 1]))
g1 = hp.massless(*np.array([np.sqrt(14), 2, 3, 1]))
g2 = hp.massless(*np.array([np.sqrt(10), 3, 0, 1]))
g3 = hp.massless(*np.array([5, 0, -3, 4]))
hcase = ['+']
ref   = hp.massless(*np.array([np.sqrt(2), 1, -1, 0]))
ttg = hp.ttg(t1, tbar2, g3, hcase, ref)

hcase1 = ['-', '-', '+']
hcase2 = ['+', '-', '-']
ggg1 = hp.ggg(g1, g2, g3, hcase1)
ggg2 = hp.ggg(g3, g1, g2, hcase2)

t1 = hp.massive(*[])
tbar2 = hp.massive(*[])
h3 = hp.massive(*[])
g4 = hp.massless(*[])
hcase = ['+']
ref1 = hp.massless(*[])
ref2 = hp.massless(*[])

tthg1 = core.tthg(t1, tbar2, h3, g4, hcase, ref1)
tthg2 = core.tthg(t1, tbar2, h3, g4, hcase, ref2)
print(tthg1)
print(tthg2)