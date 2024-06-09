import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import cm

import sys # FIXME

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
    N = len(F(guess))

    DF_mat = lambda X : np.vstack([jac(X), X.T])
    assert np.shape(DF_mat(guess)) == (N+1, N+1)


    F_mat = lambda X : np.append(np.array([func for func in F(X)]), 0)
    assert np.shape(F_mat(guess)) == (N+1,)

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


"""
Notes to self:

    # FIXME check that F satisfies needed criteria, i.e.,
        # (1) homogeneous (is there a way to do this? the functions
        #     are handed in as black box functions, so I don't think
        #     this is strictly possible?) (there's an implicit assert for this,
        #     since we check that the shifted zero is still a zero, so sort
        #     of DONE)
        # (2) n polynomials in (n+1) variables (DONE)
        # (3) anything else?

    # FIXME should F be a list of lambda functions? How else
    # might we store these? And are lambda functions the best way?
    # (Does python have other ways of writing polynomials?)
    # - Numpy does have a polynomial class, but it isn't useful for
    #   multivariable polynomials; lambda functions may be the way,
    #   so let's just assume that format for now

    # TODO should the fancier sampling algorithm from the paper be used
    # used to get the initial zeros (Lemma 4.5, z.B)

    # TODO vectorization: it might be possible to vectorize some of these
                          computations, even with all the vectors of matrices
                          floating about. If it is possible, it will likely
                          require some trickiness with axes?
      - also, logm and expm aren't vectorized; is there anything we can do here?


Next steps:
    # Figure out jac for given example as an input
    # Run with given example and check against previous result
    # Generalize building jacobian
    # Generalize finding initial zeros

"""


def main(F, zeros, jac):
    # F is the system of polynomials we are solving
    # zeros is a list of initial zeros of each poly in F (not common zeros, just plain old zeros)
    # jac is jacobian of poly system F


    """ INITIAL CHECKS """

    arg_counts = [func.__code__.co_argcount for func in F]

    # check that each func in F has the same number of variables
    assert np.allclose(arg_counts, arg_counts[0])

    # verify that this number of varialbes is exactly one more
    # than the number of polynomials in the system
    N = len(F)
    num_vars = arg_counts[0]
    assert num_vars == (N + 1)


    """ BUILD START SYSTEM """

    # TODO compute initial zeros, store in list list `zeros' (currently an input)

    # scaling initial zeros to have magnitude one
    zeros = [zero / np.linalg.norm(zero) for zero in zeros]

    # check that all scaled zeros are still zeros of the original
    # polynomial system (this is also an implicit check that the original
    # system is indeed homogeneous, as required)
    assert np.allclose([F[idx](*zeros[idx]) for idx in range(N)], 0)

    # now we use build_unitary to construct a unitary matrix that maps
    # each initial zero to the first in the list (i.e., zeros[idx] will
    # be (almost) the common zero of our start system)
    A = np.array([build_unitary(zeros[idx], zeros[0]) for idx in range(N)])
    assert np.allclose([A[idx] @ zeros[idx] for idx in range(N)], zeros[0])

    # to make the start system truly generic, we hit the matrix system A
    # with an arbitrary unitary matrix
    V = get_random_unitary(n)
    A = [V @ A[idx] for idx in range(N)]

    # check that the new polynomial system (A \cdot F) has common zero
    # V @ zeros[0]
    np.allclose([F[idx](*A[idx].T.conj() @ V @ zeros[0]) for idx in range(N)], 0)


    """ BUILD PATH """

    # now we need to build a path in \mathcal{U} that maps A \cdot F to array
    # of n by n identity matrices

    # (to do this, we are following section 3.4 in RH part 1)
    log_A = [logm(A[idx]) for idx in range(N)]

    # this choice of T might be giving us the needed Lipschitz continuity? FIXME
    T = sqrt( (1/sqrt(2)) * sum([np.linalg.norm(A[idx])**2 for idx in range(N)]))

    # build path, which is a function of time that returns a 1xN vector
    # of (unitary) matrices
    path = get_path(log_A, T)

    # let's check that this path does what we expect

    # at t = 0, path(t) should equal a 1xN vector of id(n)
    np.allclose(path(0), [np.eye(n)]*N)

    # at t = T, path(t) should equal a 1xN vector of matrices A
    np.allclose(path(T), A)


    """ TRACK ZERO """

    # next we track the common zero V @ zeros[0] along this path using a
    # projective Newton's method

    # FIXME we need to figure out how to set num_steps to guarantee convergence to
    # some precision
    num_steps = 10

    next_zero = V @ zeros[0]
    times = np.linspace(T, 0, num_steps)
    for t in times:
        W_t = path(t)
        F_t = lambda X : np.array([F[idx](*W_t[idx].T.conj() @ X) for idx in range(N)])

        # TODO build jac, which requires some knowledge of, or approximation of,
        # derivatives of F_t (currently an input, but this is wrong FIXME)

        next_zero, _ = proj_newton(next_zero, F_t, jac)

    # the final zero we find (at t = 0) is the common zero of our original system;
    # let's check this
    final_zero = next_zero
    assert np.allclose([F[idx](*final_zero) for idx in range(N)], 0)
    print('Zero of original system: ', final_zero)

    return final_zero


if __name__ == '__main__':
    # test (homogeneous) polynomials
    f = lambda x, y, z: x**2 + y**2 - z**2
    g = lambda x, y, z : 3*x**4 + y**4 - z**4

    # count number of variables
    n = 3

    # TODO TAKE OUT FIXME
    # first example (real_example_1.png)
    p = np.array([0.43070642, 0.56079585, 0.70710678])
    q = np.array([0.43565655, 0.58172984, 0.68687245])
    U = np.array([[0.98699345, -0.11835762, 0.10879064],
                  [0.12318842, 0.99162498, -0.03878815],
                  [-0.10328865, 0.0516854, 0.99330764]])
    V = np.array([[0.98003022, -0.19120783, -0.05459244],
                  [0.1618626, 0.92656294, -0.33953147],
                  [0.11550441, 0.32391462, 0.93900908]])

    # now our start system is F = (VU \cdot f, V \cdot g) with common zero
    # V @ q; let's verify this is a common zero

    # next we track the common zero V @ q along this path using a
    # projective Newton's method

    # if flag make_plot is True, we want to visualize this process. To do
    # so, we'll need to store each intermediate zero
    make_plot = False
    if make_plot: zeros = []

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

