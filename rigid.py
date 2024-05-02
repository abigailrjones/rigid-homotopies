
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# need matrix exponentials and matrix logarithms (which are concepts
# that, according to Wikipedia, lead to Lie theory; intriguing)
from scipy.linalg import logm, expm
from scipy.optimize import root


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


def proj_newton(guess, F, jac, tol=1e-8, max_iter=5000):
    # guess is a guess for where the zero is located
    # F is the polynomial system
    # jac is the Jacobian of F (should be n x (n + 1), since F has n
    # polynomials with (n + 1) variables)
    # the returned approx. zero will be within tol of the true zero
    # (if the computation succeeds, anyway)

    n = len(F(guess))
    give_up = 1e10

    DF_mat = lambda X : np.vstack([jac(X), X.T])
    assert np.shape(DF_mat(guess)) == (n+1, n+1)


    F_mat = lambda X : np.append(np.array([f for f in F(X)]), 0)
    assert np.shape(F_mat(guess)) == (n+1,)

    next_guess = guess - (np.linalg.inv(DF_mat(guess)) @ F_mat(guess))
    count = 1
    err = np.linalg.norm(next_guess - guess)
    guess = next_guess

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


def plot_shifted_variety(ax, cs, approx_zero, mat, color, label=None):
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
        new_xs.append(x)
        new_ys.append(y)

    ax.plot(new_xs, new_ys, c=color, linewidth=1.5, label=label)
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
    q = np.array([s, t, (3*s**4 + t**4)**(1/4)])       # zero of g


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
    next_zero = V @ q
    for t in np.linspace(T, 0, 100):
        W_1, W_2 = path(t)
        F_t = lambda X : np.array([f(*W_1.T.conj() @ X), g(*W_2.T.conj() @ X)])

        # we build the jacobian for the new poly system F_t for each time t
        # (note that some elements of this function are hard-coded and specific to
        # the test functions f and g FIXME)
        def jac(X):
            x1,y1,z1 = W_1.T.conj() @ X
            first_row = np.array([2*x1, 2*y1, -2*z1]) @ W_1.T.conj()

            x2,y2,z2 = W_2.T.conj() @ X
            second_row = np.array([12*x2**3, 4*y2**3, -4*z2**3]) @ W_2.T.conj()
            return np.array([first_row, second_row])

        next_zero, _ = proj_newton(next_zero, F_t, jac)

    print('Zero of original system: ', next_zero)
    print('Evaluating f at this zero: ', f(*next_zero))
    print('Evaluating g at this zero: ', g(*next_zero))

