import amplitudes.amps as amps
import tests
import amplitudes.helpers as hp

# All the color-ordered helicity amplitudes for (tth + n partons) are in amps.py.
# To obtain a helicity amplitude call the function for the desired amplitude and helicity case.
# Returns the helicity amplitude as a 2x2 matrix.
# Spin of the top quark runs along axis 0, spin of the anti-top quark runs along axis 1.
# The tthg amplitude also requires a reference momentum 'ref', which cannot be proportional to g4.
# Example for tthgg amplitude:

# phase space variables for tthgg scattering in the Higgs boson's rest frame
phi1   = 2.02
theta1 = 0.4
phi2   = 1.2
theta2 = 0.9
phi4   = 0.9
theta4 = 1.2
phi5   = 2.1
theta5 = 0.45
angles = [phi1, theta1, phi2, theta2, phi4, theta4, phi5, theta5]

# Choose a helicity case by listing the helicities of the massless particles as + or - in order
hcase = ['-', '+']

# Turn the phase-space variables to components of momenta
momenta = hp.tthggMomenta(angles)

# optionally boost the momenta:
v = 0.9
phi = 1.5
theta = 0.6
momenta_b = hp.boost(momenta, v, phi, theta)

# Evaluate color-ordered subamplitude
tthgg = amps.tthgg(*momenta_b, hcase)
print(f'tthgg for {hcase[0]}, {hcase[1]} helicities =\n{tthgg}\n')

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

# ---------------------------------------------------------------
# ttgg boost test: 

v = 0.9
phi = 1.5
theta = 0.6

tests.boost_ttgg(v, phi, theta)

# ---------------------------------------------------------------
# tthg boost test: 

v = 0.5
phi = 1.5
theta = 0.6

tests.boost_tthg(v, phi, theta)

# ---------------------------------------------------------------
# tthgg boost test: 

v = 0.95
phi = 1.3
theta = 0.8

tests.boost_tthgg(v, phi, theta)

# ---------------------------------------------------------------
# tthggg boost test: 

v = 0.7
phi = 1.3
theta = 0.2

tests.boost_tthggg(v, phi, theta)

# ---------------------------------------------------------------
# tthgggg boost test: 

v = 0.25
phi = 0
theta = 0.5

tests.boost_tthgggg(v, phi, theta)