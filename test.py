import numpy as np
import amplitudes.helpers as hp

t1    = np.array([5, 1, 1, 0])
tbar2 = np.array([5, 1, 0, 1])
g3    = np.array([np.sqrt(2), 1, 0, 1])
hcase = ['+']
ref   = np.array([np.sqrt(2), 1, -1, 0])

spin1 = hp.massive(*t1)
spin3 = hp.massless(*g3)

amp = hp.ttg(t1, tbar2, g3, hcase, ref)