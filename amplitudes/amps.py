import amplitudes.core as core
import amplitudes.helpers as hp

def tthg(t1, tbar2, h3, g4, hcase, ref):
    '''
    Color-ordered tthg amplitude. Input momenta as four-component lists and hcase = '+' or '-'.
    Outputs a 2x2 matrix with top quark spin along axis 0 and anti-top quark spin along axis 1.
    '''
    t1 = hp.massive(t1)
    tbar2 = hp.massive(tbar2)
    h3 = hp.massive(h3)
    g4 = hp.massless(g4)
    ref = hp.massless(ref)
    return core.tthg(t1, tbar2, h3, g4, hcase, ref)

def tthgg(t1, tbar2, h3, g4, g5, hcase):
    '''
    Color-ordered tthgg amplitude. Input momenta as four-component lists and hcase = ['+/-', '+/-'],
    Outputs a 2x2 matrix with top quark spin along axis 0 and anti-top quark spin along axis 1.
    '''
    t1 = hp.massive(t1)
    tbar2 = hp.massive(tbar2)
    h3 = hp.massive(h3)
    g4 = hp.massless(g4)
    g5 = hp.massless(g5)
    return core.tthgg(t1, tbar2, h3, g4, g5, hcase)

def tthqq(t1, tbar2, h3, q4, qbar5, hcase):
    '''
    Color-ordered tthqq amplitude. Input momenta as four-component lists and hcase = ['+/-', '+/-'],
    Outputs a 2x2 matrix with top quark spin along axis 0 and anti-top quark spin along axis 1.
    '''
    t1 = hp.massive(t1)
    tbar2 = hp.massive(tbar2)
    h3 = hp.massive(h3)
    q4 = hp.massless(q4)
    qbar5 = hp.massless(qbar5)
    return core.tthqq(t1, tbar2, h3, q4, qbar5, hcase)

def tthggg(t1, tbar2, h3, g4, g5, g6, hcase):
    '''
    Color-ordered tthggg amplitude. Input momenta as four-component lists and hcase = ['+/-', '+/-', '+/-'],
    Outputs a 2x2 matrix with top quark spin along axis 0 and anti-top quark spin along axis 1.
    '''
    t1 = hp.massive(t1)
    tbar2 = hp.massive(tbar2)
    h3 = hp.massive(h3)
    g4 = hp.massless(g4)
    g5 = hp.massless(g5)
    g6 = hp.massless(g6)
    return core.tthggg(t1, tbar2, h3, g4, g5, g6, hcase)

def tthqqg(t1, tbar2, h3, q4, qbar5, g6, hcase):
    '''
    Color-ordered tthqqg amplitude. Input momenta as four-component lists and hcase = ['+/-', '+/-', '+/-'],
    Outputs a 2x2 matrix with top quark spin along axis 0 and anti-top quark spin along axis 1.
    '''
    t1 = hp.massive(t1)
    tbar2 = hp.massive(tbar2)
    h3 = hp.massive(h3)
    q4 = hp.massless(q4)
    qbar5 = hp.massless(qbar5)
    g6 = hp.massless(g6)
    return core.tthqqg(t1, tbar2, h3, q4, qbar5, g6, hcase)

def tthgggg(t1, tbar2, h3, g4, g5, g6, g7, hcase):
    '''
    Color-ordered tthgggg amplitude. Input momenta as four-component lists and hcase = ['+/-', '+/-', '+/-', '+/-'],
    Outputs a 2x2 matrix with top quark spin along axis 0 and anti-top quark spin along axis 1.
    '''
    t1 = hp.massive(t1)
    tbar2 = hp.massive(tbar2)
    h3 = hp.massive(h3)
    g4 = hp.massless(g4)
    g5 = hp.massless(g5)
    g6 = hp.massless(g6)
    g7 = hp.massless(g7)
    return core.tthgggg(t1, tbar2, h3, g4, g5, g6, g7, hcase)

def tthqqgg(t1, tbar2, h3, q4, qbar5, g6, g7, hcase):
    '''
    Color-ordered tthqqgg amplitude. Input momenta as four-component lists and hcase = ['+/-', '+/-', '+/-', '+/-'],
    Outputs a 2x2 matrix with top quark spin along axis 0 and anti-top quark spin along axis 1.
    '''
    t1 = hp.massive(t1)
    tbar2 = hp.massive(tbar2)
    h3 = hp.massive(h3)
    q4 = hp.massless(q4)
    qbar5 = hp.massless(qbar5)
    g6 = hp.massless(g6)
    g7 = hp.massless(g7)
    return core.tthqqgg(t1, tbar2, h3, q4, qbar5, g6, g7, hcase)

def tthqqqq(t1, tbar2, h3, q4, qbar5, q6, qbar7, hcase):
    '''
    Color-ordered tthqqqq amplitude. Input momenta as four-component lists and hcase = ['+/-', '+/-', '+/-', '+/-'],
    Outputs a 2x2 matrix with top quark spin along axis 0 and anti-top quark spin along axis 1.
    '''
    t1 = hp.massive(t1)
    tbar2 = hp.massive(tbar2)
    h3 = hp.massive(h3)
    q4 = hp.massless(q4)
    qbar5 = hp.massless(qbar5)
    q6 = hp.massless(q6)
    qbar7 = hp.massless(qbar7)
    return core.tthqqqq(t1, tbar2, h3, q4, qbar5, q6, qbar7, hcase)