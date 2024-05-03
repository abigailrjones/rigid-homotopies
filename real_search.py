import numpy as np
from scipy.linalg import logm


def build_unitary(p, q):
    M = q[:, np.newaxis] @ p[np.newaxis]

    u, s, vt = np.linalg.svd(M)
    return u @ vt


def get_random_unitary(n):
    u, s, vt = np.linalg.svd(np.random.rand(n,n))
    return u @ vt


flag = True


while flag:
    # test (homogeneous) polynomials
    f = lambda x, y, z: x**2 + y**2 - z**2
    g = lambda x, y, z : 3*x**4 + y**4 - z**4

    # count number of variables
    n = 3

    # zeros of test polynomials
    s,t = np.random.rand(2)
    p = np.array([s, t, (s**2 + t**2)**(1/2)])       # zero of f
    s,t = np.random.rand(2)
    q = np.array([s, t, (3*s**4 + t**4)**(1/4)])       # zero of g

    # scaled zeros
    p = p / np.linalg.norm(p)
    q = q / np.linalg.norm(q)

    # unitary matrix that takes p to q
    U = build_unitary(p, q)

    # random unitary matrix to make evthg generic
    V = get_random_unitary(n)

    # we compute these to build our path
    A_1 = logm(V @ U)
    A_2 = logm(V)
    A = [A_1, A_2]

    # if we want the path to stick to matrices that are all real (so that
    # we can visualize things nicely), we need to guarantee that A_1 and
    # A_2 are real
    if np.all(np.isreal(A_1)) and np.all(np.isreal(A_2)):
        print('p: ', p)
        print('q: ', q)
        print('U: ', U)
        print('V: ', V)

        flag = False

