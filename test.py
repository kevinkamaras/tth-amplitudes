import numpy as np
import amplitudes.helpers as hp

t1    = np.array([5, 1, 1, 0])
tbar2 = np.array([5, 1, 0, 1])
g1 = np.array([np.sqrt(14), 2, 3, 1])
g2 = np.array([np.sqrt(10), 3, 0, 1])
g3 = np.array([5, 0, -3, 4])
hcase = ['+']
ref   = np.array([np.sqrt(2), 1, -1, 0])
ttg = hp.ttg(t1, tbar2, g3, hcase, ref)

spin1 = hp.massive(*t1)
spin3 = hp.massless(*g1)

hcase1 = ['-', '-', '+']
hcase2 = ['+', '-', '-']
ggg1 = hp.ggg(g1, g2, g3, hcase1)
ggg2 = hp.ggg(g3, g1, g2, hcase2)