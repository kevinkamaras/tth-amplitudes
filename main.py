import amplitudes.amps as amps
import tests

# how to obtain an amplitude:
# write the (all outgoing) four-momenta involved in the amplitude as four-component lists
# for example for tthggg:
t1    = [   5,   -3, -1,  0]
tbar2 = [   5,    0,  1, -4]
h3    = [   5,    0,  0,  1]
g4    = [-7.5,  7.5,  0,  0]
g5    = [-7.5, -7.5,  0,  0]
g6    = [   1,    0,  1,  0]

# choose a helicity case by listing the helicities of the massless particles as + or - in order
helicity_case = ['+', '-', '-']

# all the color-ordered helicity amplitudes for (tth + n partons) are in amps.py
# to obtain a helicity amplitude call the function for the desired amplitude and helicity case
# returns the helicity amplitude as a 2x2 matrix
# spin of the top quark runs along axis 0, spin of the anti-top quark runs along axis 1
tthggg_amplitude = amps.tthggg(t1, tbar2, h3, g4, g5, g6, helicity_case)
print(f'example amplitude tthggg =\n{tthggg_amplitude}\n')

# some tests that I ran to check my amplitudes:

# -----------------------------------------------------------------
# ttqqg test: here I compare the ttqqg (no Higgs boson) amplitude I get for helicity case [-, +, +]
# with the explicit expression found in the Campbell and Ellis paper at the bottom of page 12
# this tests that the BCFW shift and on-shell condition are programmed correctly
# also tests the subamplitdes ttgg, ttqq, ttg, qqg
t1    = [   5,   -3, -1,  0]
tbar2 = [   5,    0,  1, -4]
q3    = [   5,    3,  0,  4]
qbar4 = [-7.5,  7.5,  0,  0]
g5    = [-7.5, -7.5,  0,  0]
momenta = [t1, tbar2, q3, qbar4, g5]
tests.ttqqg_test(momenta)
# ---------------------------------------------------------------



