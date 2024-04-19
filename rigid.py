
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# need matrix exponentials and matrix logarithms (which are concepts
# that, according to Wikipedia, lead to Lie theory; intriguing)
from scipy.linalg import logm, expm


def build_unitary(p, q):
    # this process uses the SVD and stems from the solution to the
    # balanced Procrustes problem (see Wikipedia page for more details)

    # we have to mess about with axes here to make the matrix multiplication
    # work as expected; the reason we don't write p and q in this form from
    # the beginning is so that evaluating f and g at these points is very
    # straightforward
    M = q[:, np.newaxis] @ p[np.newaxis]

    # a quick check to see that M looks as expect HARD-CODED, FIXME
    assert M.shape == (3,3)

    u, s, vt = np.linalg.svd(M)
    return u @ vt


def get_random_unitary(n):
    u, s, vt = np.linalg.svd(np.random.rand(n,n))
    return u @ vt


def get_path(A, T):
    w_t = lambda t : [expm((t/T) * A_j) for A_j in A]
    return w_t


if __name__ == '__main__':
    # test (homogeneous) polynomials
    f = lambda x, y, z: x**2 + y**2 - z**2
    g = lambda x, y, z: x + y - z

    # count number of variables
    n = 3

    # zeros of test polynomials (found manually FIXME)
    p = np.array([1, 1, sqrt(2)]) # zero of f
    q = np.array([1, 1, 2])       # zero of g


    # scaled zeros of test polynomials
    p = p / np.linalg.norm(p)
    q = q / np.linalg.norm(q)


    # both zeros should now have magnitude 1,
    assert np.linalg.norm(p) == np.linalg.norm(q)


    # both zeros should also still satisfy their respective polynomials
    # (since both polynomials are homogeneous)
    assert np.isclose(f(*p), 0)
    assert np.isclose(g(*q), 0)


    # now we use build_unitary to construct a unitary matrix that maps
    # p to q
    U = build_unitary(p, q)


    # check that unitary matrix does what we expect it to do
    assert np.allclose(q, U @ p)


    # now U \cdot f = f (U^(-1) @ q) = f(p) = 0 and also g (q) = 0, so q is
    # a common zero of the polynomial system F = (U \cdot f, g)


    # to make the start system truly generic, we hit this polynomial system
    # by an arbitrary unitary matrix
    V = get_random_unitary(n)


    # now our start system is F = (VU \cdot f, V \cdot g) with common zero
    # V @ q; let's verify this is a common zero
    np.isclose(f(*U.T @ V.T @ V @ q), 0)
    np.isclose(g(*V.T @ V @ q), 0)


    # now we need to build a path in \mathcal{U} that maps (VU, U) to
    # (id(n), id(n)), where id(n) is the n by n identity matrix

    # (to do this, we are following section 3.4 in RH part 1)

    # HARD-CODED, FIXME
    A_1 = logm(V @ U)
    A_2 = logm(U)
    A = [A_1, A_2]

    # this choice of T might be giving us the needed Lipschitz continuity?
    T = sqrt( (1/sqrt(2)) * sum([np.linalg.norm(A_j) for A_j in A]) )
    path = get_path(A, T)


    # let's check that this path does what we expect

    # at t = 0, path(t) should equal (id(n), id(n))
    np.allclose(path(0), (np.eye(n), np.eye(n)))

    # at t = T, path(t) should equal (VU, U)
    np.allclose(path(T), (V@U, U))


    # next we track the common zero V @ q along this path using a
    # predictor-corrector method

    # (a sort of sneaky thing we also need to do is set up the system of DEs)

