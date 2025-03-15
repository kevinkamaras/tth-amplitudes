import amplitudes.amps as amps
import tests

# How to obtain an amplitude:
# Write the (all outgoing) four-momenta involved in the amplitude as four-component lists.
# For example for tthggg:
t1    = [   5,   -3, -1,  0]
tbar2 = [   5,    0,  1, -4]
h3    = [   5,    0,  0,  1]
g4    = [-7.5,  7.5,  0,  0]
g5    = [-7.5, -7.5,  0,  0]
g6    = [   1,    0,  1,  0]

# Choose a helicity case by listing the helicities of the massless particles as + or - in order
helicity_case = ['+', '-', '-']

# All the color-ordered helicity amplitudes for (tth + n partons) are in amps.py.
# To obtain a helicity amplitude call the function for the desired amplitude and helicity case.
# Returns the helicity amplitude as a 2x2 matrix.
# Spin of the top quark runs along axis 0, spin of the anti-top quark runs along axis 1.
tthggg_amplitude = amps.tthggg(t1, tbar2, h3, g4, g5, g6, helicity_case)

print(f'\nexample amplitude tthggg =\n{tthggg_amplitude}\n')


# Some tests that I ran to check my amplitudes:
# -----------------------------------------------------------------
# ttqqg test: Here I compare the ttqqg (no Higgs boson) amplitude I get for helicity case [-, +, +]
#             with the explicit expression found in the Campbell and Ellis paper at the bottom of page 12.
#             This tests that the BCFW shift and on-shell condition are programmed correctly.
#             Also tests the subamplitdes ttgg, ttqq, ttg, qqg.
t1    = [   5,   -3, -1,  0]
tbar2 = [   5,    0,  1, -4]
q3    = [   5,    3,  0,  4]
qbar4 = [-7.5,  7.5,  0,  0]
g5    = [-7.5, -7.5,  0,  0]
momenta = [t1, tbar2, q3, qbar4, g5]

tests.ttqqg_test(momenta)

# ---------------------------------------------------------------
# ttgg test: Here I compare the ttgg amplitude squared summed over color and spins
#            with the expression for the cross section found at the end of Gluck1978.
#            This test makes sure that the massive and massless spinors are programmed correctly
#            and that the matrix multiplications correspond to the correct index contractions.
#            I work in the center of mass frame of the two massive quarks.
#            The degrees of freedom are the two angles for the top pair,
#            the momentum of the top pair, and the two angles for the gluons. 

phiTop = 1.27
thetaTop = 0.78
pTop = 200
phiGlu = 0.97
thetaGlu = 2.5

tests.ttgg_test(phiTop, thetaTop, pTop, phiGlu, thetaGlu)

# ---------------------------------------------------------------
# tthg test: Here I compare the tthg amplitude for four different reference spinors.
#            This test checks that the amplitudes really are independent of the reference spinor.
#            I work in the center of mass frame of the two massive quarks.
#            The degrees of freedom are the angles and momentum for the top pair
#            and the angles for the Higgs and gluon.
#            The momenta I use for the reference spinors are:
#               ref1 = [1, 1,  0, 0]
#               ref2 = [1, 0,  1, 0]
#               ref3 = [1, 0,  0, 1]
#               ref4 = [5, 3, -4, 0]

phiTop = 1.27
thetaTop = 0.78
pTop = 200
phiGlu = 0.97
thetaGlu = 2.5
hcase = '+'

tests.tthg_test(phiTop, thetaTop, pTop, phiGlu, thetaGlu, hcase)