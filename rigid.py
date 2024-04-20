
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
    f = lambda x, y, z: x**6 + y**6 - z**6
    g = lambda x, y, z : 3*x**5 + y**5 - z**5

    # count number of variables
    n = 3


    # zeros of test polynomials (found manually FIXME)
    s,t = np.random.rand(2)
    p = np.array([s, t, (s**6 + t**6)**(1/6)])       # zero of f
    s,t = np.random.rand(2)
    q = np.array([s, t, (3*s**5 + t**5)**(1/5)])       # zero of g


    # scaled zeros of test polynomials
    p = p / np.linalg.norm(p)
    q = q / np.linalg.norm(q)


    # both zeros should now have magnitude 1,
    assert np.isclose(np.linalg.norm(p), np.linalg.norm(q))


    # both zeros should also still satisfy their respective polynomials
    # (since both polynomials are homogeneous)
    assert np.isclose(f(*p), 0)
    assert np.isclose(g(*q), 0)


    # now we use build_unitary to construct a unitary matrix that maps
    # p to q
    U = build_unitary(p, q)


    # check that unitary matrix does what we expect it to do
    assert np.allclose(q, U @ p)




    # an exercise in plotting FIXME

    # TODO
    # 1. add an axes argument that we can pass to a plotting function
    # 2. refactor the variety shift bit into its own function (so it can be
    #    easily reused)
    # 3. generally improve labeling and ordering and comments
    # 4. I don't fully understanding the fixing z part. In particular, I am
    #    fixing z = p[-1] for the f variety and z = q[-1] for the g variety.
    #    But then when I shift the f variety, I set... z = p[-1], which is
    #    exactly what it's supposed to be... Not sure why I found that confusing,
    #    but it is the right choice (according to a check of other values), so
    #    I apparently understood something when I was doing it. (make a comment about this)

    print('Determinant of U: ', np.linalg.det(U))

    step = 100
    ran = 2
    xs = np.linspace(-ran,ran,step)
    ys = np.linspace(-ran,ran,step)
    X,Y = np.meshgrid(xs, ys)

    plt.axis('equal')
    plt.scatter(*p[0:-1], c='k')
    plt.scatter(*q[0:-1], c='k')
    # make order standard here FIXME
    cs_f = plt.contour(X, Y, f(X,Y,p[-1]), levels=[0], colors='slateblue', alpha=0.45, algorithm='serial', linewidths=0.9)
    cs_g = plt.contour(X, Y, g(X,Y,q[-1]), levels=[0], colors='k', alpha=1, linewidths=0.95, algorithm='serial')
    # plt.show()

    # collections is deprecated in matplotlib version 3.8, but
    # I'm running 3.7.5 FIXME
    new_xs = []
    new_ys = []
    count = 1
    for vertex, code in cs_f.collections[0].get_paths()[0].iter_segments():
        x,y = (U @ np.append(vertex,p[-1]))[0:-1]
        if count == 1:
            plt.scatter(*vertex, c='red')
            plt.scatter(x,y, c='r')
            count += 1
        new_xs.append(x)
        new_ys.append(y)

    plt.plot(new_xs, new_ys, c='slateblue', linewidth=2)

    # plt.legend()
    plt.show()




    # now (U \cdot f) (q) = f (U^(-1) @ q) = f(p) = 0 and also g (q) = 0, so q is
    # a common zero of the polynomial system F = (U \cdot f, g)


    # to make the start system truly generic, we hit this polynomial system
    # with an arbitrary unitary matrix
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

