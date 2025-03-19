import amplitudes.amps as amps
import tests
import numpy as np

# How to obtain an amplitude:
# Write the (all outgoing) four-momenta involved in the amplitude as four-component lists:
t1    = [   5,   -3, -1,  0]
tbar2 = [   5,    0,  1, -4]
h3    = [   5,    0,  0,  1]
g4    = [-7.5,  7.5,  0,  0]
g5    = [-7.5, -7.5,  0,  0]
g6    = [   1,    0,  1,  0]
g7    = [   1,    0, -1,  0]

# Choose a helicity case by listing the helicities of the massless particles as + or - in order
hcase = ['-', '+', '-', '+']

# All the color-ordered helicity amplitudes for (tth + n partons) are in amps.py.
# To obtain a helicity amplitude call the function for the desired amplitude and helicity case.
# Returns the helicity amplitude as a 2x2 matrix.
# Spin of the top quark runs along axis 0, spin of the anti-top quark runs along axis 1.
# The tthg amplitude also requires a reference momentum 'ref', which cannot be proportional to g4.
# Example for tthgggg amplitude:

# top and Higgs masses
mTop = 171
mHiggs = 125

# phase space variables for tthgggg scattering in the Higgs boson's rest frame
phiTop = 1.02
thetaTop = 0.87
pTop = 200
phiGlu1 = 1.2
thetaGlu1 = 0.9
phiGlu2 = 0.31
thetaGlu2 = 1

# Turn the phase-space variables to components of momenta
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

# Initialize four-momenta for tthgggg scattering
t1    = np.array([Etop, ptopx, ptopy, ptopz])
tbar2 = np.array([Etop, -ptopx, -ptopy, -ptopz])
h3    = np.array([mHiggs, 0, 0, 0])
g4    = np.array([-Eglu, pglux1, pgluy1, pgluz1])
g5    = np.array([-Eglu, pglux2, pgluy2, pgluz2])
g6    = np.array([-Eglu, -pglux1, -pgluy1, -pgluz1])
g7    = np.array([-Eglu, -pglux2, -pgluy2, -pgluz2])

# Evaluate color-ordered subamplitude
# tthgggg = amps.tthgggg(t1, tbar2, h3, g4, g5, g6, g7, hcase)

# print(f'tthgggg for {hcase[0]}, {hcase[1]}, {hcase[2]}, {hcase[3]} helicities =\n{tthgggg}\n')


# Some tests that I ran to check my amplitudes:
# -----------------------------------------------------------------
# ttqqg test: Here I compare the ttqqg (no Higgs boson) amplitude I get for helicity case [-, +, +]
#             with the explicit expression found in the Campbell and Ellis paper at the bottom of page 12.
#             This tests that the BCFW shift and on-shell condition are programmed correctly.
#             Also tests the subamplitdes ttgg, ttqq, ttg, qqg.
t1    = [   15,   -1.5, -1, -2]
tbar2 = [   15,   -1.5,  1, -2]
q3    = [    5,    3,  0,  4]
qbar4 = [-17.5,  17.5,  0,  0]
g5    = [-17.5, -17.5,  0,  0]
momenta = [t1, tbar2, q3, qbar4, g5]

# uncomment to run test:
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

# uncomment to run test:
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

# uncomment to run test:
tests.tthg_test(phiTop, thetaTop, pTop, phiGlu, thetaGlu, hcase)

# ---------------------------------------------------------------
# ttgg boost test: 

v = 0.9
phi = 1.5
theta = 0.6

bx = np.sin(theta) * np.cos(phi)
by = np.sin(theta) * np.sin(phi)
bz = np.cos(theta)

beta = v * np.array([bx, by, bz])

# uncomment to run test:
tests.boost_ttgg(beta)

# ---------------------------------------------------------------
# tthg boost test: 

v = 0.9999
phi = 1.5
theta = 0.6

bx = np.sin(theta) * np.cos(phi)
by = np.sin(theta) * np.sin(phi)
bz = np.cos(theta)

beta = v * np.array([bx, by, bz])

# uncomment to run test:
tests.boost_tthg(beta)

# ---------------------------------------------------------------
# tthgg boost test: 

v = 0.999
phi = 1.5
theta = 0.6

bx = np.sin(theta) * np.cos(phi)
by = np.sin(theta) * np.sin(phi)
bz = np.cos(theta)

beta = v * np.array([bx, by, bz])

# uncomment to run test:
tests.boost_tthgg(beta)

# ---------------------------------------------------------------
# tthgggg boost test: 

v = 0.9
phi = 1.5
theta = 0.6

bx = np.sin(theta) * np.cos(phi)
by = np.sin(theta) * np.sin(phi)
bz = np.cos(theta)

beta = v * np.array([bx, by, bz])

# uncomment to run test:
tests.boost_tthgggg(beta)