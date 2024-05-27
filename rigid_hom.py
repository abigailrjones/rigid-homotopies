import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import cm

# need matrix exponentials and matrix logarithms (which are concepts
# that, according to Wikipedia, lead to Lie theory; intriguing)
from scipy.linalg import logm, expm


def build_unitary(moved_zero, fixed_zero):
    # this process uses the SVD and stems from the solution to the
    # balanced Procrustes problem (see Wikipedia page for more details)

    # we mess about with axes here to make the matrix multiplication
    # work as expected
    M = fixed_zero[:, np.newaxis] @ moved_zero[np.newaxis]

    u, s, vt = np.linalg.svd(M)
    return u @ vt


def get_random_unitary(n):
    u, s, vt = np.linalg.svd(np.random.rand(n,n))
    return u @ vt


def get_path(A, T):
    w_t = lambda t : [expm((t/T) * A_j) for A_j in A]
    return w_t


def proj_newton(guess, F, jac, tol=1e-10, max_iter=5000):
    # guess is a guess for where the zero is located
    # F is the polynomial system
    # jac is the Jacobian of F (should be n x (n + 1), since F has n
    # polynomials with (n + 1) variables)
    # the returned approx. zero will be within tol of the true zero
    # (if the computation succeeds, anyway)

    # n is the number of polynomials in the system F
    n = len(F(guess))

    DF_mat = lambda X : np.vstack([jac(X), X.T])
    assert np.shape(DF_mat(guess)) == (n+1, n+1)


    F_mat = lambda X : np.append(np.array([func for func in F(X)]), 0)
    assert np.shape(F_mat(guess)) == (n+1,)

    next_guess = guess - (np.linalg.inv(DF_mat(guess)) @ F_mat(guess))
    err = np.linalg.norm(next_guess - guess)
    guess = next_guess
    count = 1
    give_up = 1e10

    while err > tol and err < give_up and count < max_iter:
        next_guess = guess - (np.linalg.inv(DF_mat(guess)) @ F_mat(guess))
        count += 1
        err = np.linalg.norm(next_guess - guess)
        guess = next_guess

    if count >= max_iter:
        raise Exception('Too many iterations.')
    if err >= give_up:
        raise Exception('Going to infinity and beyond.')

    return [guess, count]


def plot_shifted_variety(ax, X, Y, fixed_z, mats, color):
    # we use contour from matplotlib to visualize the shifted varieties
    # as t moves from T to 0

    # TODO visualization is currently VERY hard-coded to a specific set of
    # examples; this function will NOT work with almost all examples
    W_1, W_2 = mats
    xx = X.flatten()
    yy = Y.flatten()
    f_vals = []
    g_vals = []
    for idx in range(len(xx)):
        f_vals.append(f(*W_1.T.conj() @ [xx[idx], yy[idx], fixed_z]))
        g_vals.append(g(*W_2.T.conj() @ [xx[idx], yy[idx], fixed_z]))

    f_vals = np.reshape(f_vals, X.shape)
    g_vals = np.reshape(g_vals, X.shape)

    ax.contour(X, Y, f_vals, levels=[0], colors=[color], algorithm='serial')
    ax.contour(X, Y, g_vals, levels=[0], colors=[color], algorithm='serial')

    return


if __name__ == '__main__':
    # test (homogeneous) polynomials
    f = lambda x, y, z: x**2 + y**2 - z**2
    g = lambda x, y, z : 3*x**4 + y**4 - z**4

    # count number of variables
    n = 3

    # zeros of test polynomials
    s,t = np.random.rand(2)
    p = np.array([s, t, (s**2 + t**2)**(1/2)])       # zero of f
    s,t = np.random.rand(2)
    q = np.array([s, t, (3*s**4 + t**4)**(1/4)])     # zero of g


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

    # (Note that, instead using the inverse, we're taking the conjugate transpose.
    # Since U,V are unitary, these are equivalent. Also, since U,V are real, we
    # only need to transpose. Later in the code, when the matrices can become complex,
    # we'll need to conjugate too.)
    np.isclose(f(*U.T @ V.T @ V @ q), 0)
    np.isclose(g(*V.T @ V @ q), 0)


    # now we need to build a path in \mathcal{U} that maps (VU, U) to
    # (id(n), id(n)), where id(n) is the n by n identity matrix

    # (to do this, we are following section 3.4 in RH part 1)
    A_1 = logm(V @ U)
    A_2 = logm(V)
    A = [A_1, A_2]

    # this choice of T might be giving us the needed Lipschitz continuity?
    T = sqrt( (1/sqrt(2)) * sum([np.linalg.norm(A_j)**2 for A_j in A]) )

    path = get_path(A, T)


    # let's check that this path does what we expect

    # at t = 0, path(t) should equal (id(n), id(n))
    np.allclose(path(0), (np.eye(n), np.eye(n)))

    # at t = T, path(t) should equal (VU, V)
    np.allclose(path(T), (V@U, V))


    # next we track the common zero V @ q along this path using a
    # projective Newton's method

    # if flag make_plot is True, we want to visualize this process. To do
    # so, we'll need to store each intermediate zero
    make_plot = False
    if make_plot: zeros = []

    # the number of steps we'll take to get from time 0 to time T
    N = 10

    next_zero = V @ q
    times = np.linspace(T, 0, N)
    for t in times:
        W_1, W_2 = path(t)
        F_t = lambda X : np.array([f(*W_1.T.conj() @ X), g(*W_2.T.conj() @ X)])

        # we build the jacobian for the new poly system F_t for each time t
        # (note that some elements of this function are hard-coded and specific to
        # the test functions f and g FIXME)
        def jac(X):
            x1, y1, z1 = W_1.T.conj() @ X
            first_row = np.array([2*x1, 2*y1, -2*z1]) @ W_1.T.conj()

            x2, y2, z2 = W_2.T.conj() @ X
            second_row = np.array([12*x2**3, 4*y2**3, -4*z2**3]) @ W_2.T.conj()
            return np.array([first_row, second_row])

        next_zero, _ = proj_newton(next_zero, F_t, jac)
        if make_plot:
            zeros.append(next_zero)

    # the final zero we find (at t = 0) is the common zero of our original system;
    # let's check this
    final_zero = next_zero
    assert np.isclose(f(*final_zero), 0)
    assert np.isclose(g(*final_zero), 0)
    print('Zero of original system: ', final_zero)


    if make_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # these values are specific to the hard-coded example above
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)

        step = 200
        ran = 2
        xs = np.linspace(-ran,ran,step)
        ys = np.linspace(-ran,ran,step)
        X,Y = np.meshgrid(xs, ys)

        colors = cm.rainbow(np.linspace(0, 1, N))
        for idx in range(0,N):
            print(idx)
            t = times[idx]
            color = colors[idx]
            zero = zeros[idx]

            W_1, W_2 = path(t)

            label = None
            if idx == 0:
                label = 'Start system'
            if idx == (N-1):
                label = 'Original system'

            plot_shifted_variety(ax, X, Y, zero[-1], [W_1, W_2], color)
            ax.scatter(*zero[0:-1], color=color, s=40, zorder=10, edgecolors='k', label=label)

        plt.legend()
        plt.savefig('images/real_example.png')
        plt.show()

