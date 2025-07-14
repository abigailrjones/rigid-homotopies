import mpmath as mp


def scale(zero):
    return zero / mp.norm(zero)


def mean(vals):
    return sum(vals) / len(vals)


def compute_shifted_system(F, W_t, num_funcs):
    return [_shift(F[idx], W_t[idx]) for idx in range(num_funcs)]


def _shift(func, mat):
    return lambda *args: func(*mat.conjugate().T @ mp.matrix(args))


def allclose(arr1, arr2):
    if (arr1.rows != arr2.rows) or (arr1.cols != arr2.cols):
        raise ValueError("Arrays have conflicting dimensions.")

    for row_idx in range(arr1.rows):
        for col_idx in range(arr1.cols):
            if not mp.almosteq(arr1[row_idx,col_idx],
                               arr2[row_idx,col_idx],
                               rel_eps=None, abs_eps=None):
                return False

    return True


def build_jacobian(F, num_funcs, num_vars, projective=False):
    jac = list()
    for row_idx in range(num_funcs):
        row = list()
        func = F[row_idx]
        for col_idx in range(num_vars):
            row.append(_partial(func, col_idx, num_vars))
        jac.append(row)

    if projective:
        row = list()
        for col_idx in range(num_vars):
            row.append(_identity_row(col_idx))
        jac.append(row)

    return lambda X: mp.matrix([[func(X) for func in row] for row in jac])


def _partial(func, var_idx, num_vars):
    partial_wrt = [0 if i != var_idx else 1 for i in range(num_vars)]
    return lambda X: mp.diff(func, X, partial_wrt)


def _identity_row(var_idx):
    return lambda X: X[var_idx]


def pseudoinverse(mat):
    U, S, Vh = mp.svd(mat)
    Sp = [mp.mpf('1')/sing_val if not mp.almosteq(sing_val, mp.mpf('0.')) else
          mp.mpf('0.') for sing_val in S]
    return Vh.conjugate().T @ mp.diag(Sp) @ U.conjugate().T


def eval_sys(F_t, point):
    return mp.matrix([func(*point) for func in F_t])


def print_input(t, dt, projective):
    print(f"Starting at t={t} with initial stepsize {dt}.")
    if projective:
        print("Note that Newton's method will be using the projective method to make the system square.\n")
    else:
        print("Note that Newton's method will be using the pseudoinverse.\n")

    return


def print_output(F, final_zero, num_iter, avg_step_size):
    print(f"Converged in {num_iter} iterations")
    print(f"Average timestep: {avg_step_size}\n")
    print(f"Final zero:\n{mp.nstr(mp.chop(final_zero), 10)}\n")
    print(f"System residuals:\n{mp.nstr(eval_sys(F, final_zero), 10)}")

    return
