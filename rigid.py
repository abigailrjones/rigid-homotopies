
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

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
    # HARD CODED derivatives, FIXME
    f = lambda x, y, z: x**2 + y**2 - z**2
    df_x = lambda x, y, z: 2*x
    df_y = lambda x, y, z: 2*y

    g = lambda x, y, z : 3*x**8 + y**8 - z**8
    dg_x = lambda x, y, z: 24*x**7
    dg_y = lambda x, y, z: 8*y**7

    # count number of variables
    n = 3


    # zeros of test polynomials (found manually FIXME)
    s,t = np.random.rand(2)
    p = np.array([s, t, (s**2 + t**2)**(1/2)])       # zero of f
    s,t = np.random.rand(2)
    q = np.array([s, t, (3*s**8 + t**8)**(1/8)])       # zero of g

    # p = [0.89893741, 0.48690801, 1.02233452]
    # q = [0.70364412, 0.49015276, 0.80907222]

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
    A_2 = logm(V)
    A = [A_1, A_2]

    # this choice of T might be giving us the needed Lipschitz continuity?
    T = sqrt( (1/sqrt(2)) * sum([np.linalg.norm(A_j) for A_j in A]) )

    path = get_path(A, T)


    # let's check that this path does what we expect

    # at t = 0, path(t) should equal (id(n), id(n))
    np.allclose(path(0), (np.eye(n), np.eye(n)))

    # at t = T, path(t) should equal (VU, V)
    np.allclose(path(T), (V@U, V))


    # next we track the common zero V @ q along this path using a
    # predictor-corrector method

    # (a sort of sneaky thing we also need to do is set up the system of DEs)





    # an exercise in plotting FIXME

    # TODO
    # 1. add an axes argument that we can pass to a plotting function
    # 2. refactor the variety shift bit into its own function (so it can be
    #    easily reused)
    # 3. generally improve labeling and ordering and comments
    # 4. I don't fully understand the fixing z part. In particular, I am
    #    fixing z = p[-1] for the f variety and z = q[-1] for the g variety.
    #    But then when I shift the f variety, I set... z = p[-1], which is
    #    exactly what it's supposed to be... Not sure why I found that confusing,
    #    but it is the right choice (according to a check of other values), so
    #    I apparently understood something when I was doing it. (make a comment about this)
    # 5. Also, the algorithm being used to build the contour plots seems intriguing and
    #    not even remotely obvious. Working at Matplotlib (or some other similar place)
    #    would be pretty fun I think. I want to make a comment somewhere about using 'serial'
    #    for the contour algorithm, since it isn't the default in Matplotlib, which is a
    #    little silly, especially since it made everything so much better when plotting.

    def plot_shifted_variety(ax, cs, approx_zero, mat, color):
        # ax is the figure we're plotting in
        # cs is the object returned by plt.contour
        # approx_zero is the approximate zero we're working with; the final coordinate
        # is the fixed value of z we choose when plotting
        # mat is the matrix we're shifting the variety by

        # collections is deprecated in matplotlib version 3.8, but
        # I'm running 3.7.5 FIXME
        new_xs = []
        new_ys = []
        # count = 1
        for vertex, code in cs.collections[0].get_paths()[0].iter_segments():
            x,y = (mat @ np.append(vertex,approx_zero[-1]))[0:-1]
#            if count == 1:
#                plt.scatter(*vertex, c='red')
#                plt.scatter(x,y, c='r')
#                count += 1
            new_xs.append(x)
            new_ys.append(y)

        plt.plot(new_xs, new_ys, c=color, linewidth=1.5)
        return


    print('Determinant of U: ', np.linalg.det(U))

    step = 1000
    ran = 2 # FIXME
    xs = np.linspace(-ran,ran,step)
    ys = np.linspace(-ran,ran,step)
    X,Y = np.meshgrid(xs, ys)

    ax = plt.figure()
    plt.axis('equal')
#    plt.scatter(*p[0:-1], c='k')
#    plt.scatter(*q[0:-1], c='k')
    # make order standard here FIXME
    cs_f = plt.contour(X, Y, f(X,Y,p[-1]), levels=[0], colors='slateblue', alpha=0.45, algorithm='serial', linewidths=0.9)
    cs_g = plt.contour(X, Y, g(X,Y,q[-1]), levels=[0], colors='k', alpha=0.45, linewidths=0.9, algorithm='serial')


    N = 22
    colors = iter(cm.rainbow(np.linspace(0, 1, N)))
    for t in np.linspace(T, 0, N):
    # for t in np.linspace(T,0,10):
        A_1, A_2 = path(t)
        # assert np.allclose(A_1, V@U)
        # assert np.allclose(A_2, V)
        color = next(colors)
        plot_shifted_variety(ax, cs_f, p, A_1, color=color)
        plot_shifted_variety(ax, cs_g, q, A_2, color=color)
        if t == T:
            # plt.scatter(*(A_1@p)[0:-1], c='gold', s=100)
            # plt.scatter(*(A_2@q)[0:-1], c='k')
            pass


    # plt.legend()
    plt.show()


