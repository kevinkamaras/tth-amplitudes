import numpy as np
import amplitudes.helpers as hp
import amplitudes.core as core
import amplitudes.amps as amps

def ttqqg_test(momenta):
     t1 = hp.massive(momenta[0])
     tbar2 = hp.massive(momenta[1])
     q3 = hp.massless(momenta[2])
     qbar4 = hp.massless(momenta[3])
     g5 = hp.massless(momenta[4])
     hcase = ['-', '+', '+']
     ttqqg = core.ttqqg(t1, tbar2, q3, qbar4, g5, hcase)
     
     d1 = 1j * (((t1.sbra @ (qbar4.sketabra() + g5.sketabra() ) @ t1.momentum() @ g5.sket
      + hp.sbraket(t1, g5) @ hp.abraket(qbar4, g5) @ hp.sbraket(g5, qbar4)) @ hp.abraket(q3, tbar2)
      + hp.abraket(t1, q3) @ g5.sbra @ t1.sketabra() @ (qbar4.momentum() + g5.momentum()) @ tbar2.sket)
      * (qbar4.abra @ t1.momentum() @ g5.sket) / ((g5.abra @ t1.momentum() @ g5.sket)
                                                * hp.abraket(qbar4, g5) * hp.abraket(q3, qbar4) *
                                                (q3.sbra @ (qbar4.sketabra() + g5.sketabra()) @ t1.momentum() @ g5.sket)))

     p345 = q3.vector + qbar4.vector + g5.vector
     s345 = hp.minkowski(p345)
     d2 = 1j * ((t1.mass * hp.sbraket(q3, g5) * hp.sbraket(qbar4, g5)**2 * hp.abraket(t1, tbar2)) 
               / (s345 * hp.sbraket(q3, qbar4) * (q3.sbra @ (qbar4.sketabra() + g5.sketabra()) @ t1.momentum() @ g5.sket)))
     
     diff = ttqqg - (d1 + d2)
     if diff.all() <= 1:
          print('                ------------------------------')
          print('test for ttqqg passed!\n')
          print(f'amplitude from my ttqqg function:\n{ttqqg}\n')
          print(f'amplitude from explicit expression in Campbell2023:\n{d1 + d2}\n')
          print(f'diff =\n{diff}')
          print('                ------------------------------')
     return

m = 171

phitop = 1.27
thetatop = 0.78
ptop = 200
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

t1    = hp.massive([Etop, ptopx, ptopy, ptopz])
tbar2 = hp.massive([Etop, -ptopx, -ptopy, -ptopz])
g3    = hp.massless([-Eglu, pglux, pgluy, pgluz])
g4    = hp.massless([-Eglu, -pglux, -pgluy, -pgluz])
hcases = [['+', '+'],
          ['-', '-'],
          ['+', '-'],
          ['-', '+']]

hcases2 = [[hcase[1], hcase[0]] for hcase in hcases]

amps1234 = [core.ttgg(t1, tbar2, g3, g4, hcase) for hcase in hcases]
amps1243 = [core.ttgg(t1, tbar2, g4, g3, hcase) for hcase in hcases2]
subleadings = [amp1234 + amp1243 for amp1234, amp1243 in zip(amps1234, amps1243)]

N = 3
V = N**2 - 1
colorSum = []
for A1, A2, A3 in zip(amps1234, amps1243, subleadings):
     colorSum += [V * (N * (abs(A1**2) + abs(A2**2)) - (1 / N) * abs(A3**2))]

spinSum = sum([np.sum(i) for i in colorSum])

#from Gluck1978
p12 = -t1.vector - tbar2.vector
p13 = -t1.vector - g3.vector
p14 = -t1.vector - g4.vector

s = hp.minkowski(p12)
t = hp.minkowski(p13)
u = hp.minkowski(p14)

Mss = (4 / s**2) * (t - m**2) * (u - m**2)
Mtt = (-2 / (t - m**2)**2) * (4 * m**4 - (t - m**2) * (u - m**2) + 2 * m**2 * (t - m**2))
Muu = (-2 / (u - m**2)**2) * (4 * m**4 - (u - m**2) * (t - m**2) + 2 * m**2 * (u - m**2))
Mst = (4 / (s * (t - m**2))) * (m**4 - t * (s + t))
Msu = (4 / (s * (u - m**2))) * (m**4 - u * (s + u))
Mtu = ((-4 * m**2) / ((t - m**2) * (u - m**2))) * (4 * m**2 + (t - m**2) + (u - m**2))

result = (12 * Mss + (16 / 3) * Mtt + (16 / 3) * Muu + 6 * Mst + 6 * Msu - (2 / 3) * Mtu) * 4

ratio = spinSum / result
# print(f'<|M|^2> = {spinSum}')
# print(f'<|M|^2> = {result}')
# print(f'ratio = {ratio}')

mtop = 171
mhig = 125

phitop = 1.27
thetatop = 0.78
ptop = 200
phiglu = 0.97
thetaglu = 2.5

Etop = np.sqrt(ptop**2 + m**2)
ptopx = ptop * np.sin(thetatop) * np.cos(phitop)
ptopy = ptop * np.sin(thetatop) * np.sin(phitop)
ptopz = ptop * np.cos(thetatop)

Eglu = Etop - (mhig**2 / (4 * Etop))
pglux = Eglu * np.sin(thetaglu) * np.cos(phiglu)
pgluy = Eglu * np.sin(thetaglu) * np.sin(phiglu)
pgluz = Eglu * np.cos(thetaglu)

Ehig = np.sqrt(Eglu**2 + mhig**2)
t1    = hp.massive([Etop, ptopx, ptopy, ptopz])
tbar2 = hp.massive([Etop, -ptopx, -ptopy, -ptopz])
h3    = hp.massive([-Ehig, pglux, pgluy, pgluz])
g4    = hp.massless([-Eglu, -pglux, -pgluy, -pgluz])
ref   = hp.massless([1, 1, 0, 0])
hcase = '-'

tthg = core.tthg(t1, tbar2, h3, g4, hcase, ref)
# print(f'tthg =\n{tthg}')