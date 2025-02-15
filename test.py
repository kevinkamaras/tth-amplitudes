import numpy as np
import amplitudes.helpers as hp

t1    = hp.massive(5, 1, 1, 0)
tbar2 = hp.massive(5, 1, 0, 1)
g3    = hp.massless(np.sqrt(2), 1, 0, 1)
hcase = ['+']
ref   = hp.massless(np.sqrt(2), 1, -1, 0)

amp = hp.ttg(t1, tbar2, g3, hcase, ref)