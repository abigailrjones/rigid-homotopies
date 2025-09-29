include("utils.jl")

# input a vector containing a zero of each polynomial in the system, returns
# the associated start system and start root
function build_start_system(F, init_roots, num_vars)
    V = build_random_unitary(num_vars)
    start_root = V * (init_roots[1] / sum(abs.(init_roots[1]).^2))
    start_system = [V]
    for idx in 2:length(init_roots)
        root = init_roots[idx]
        scaled_root = root / sum(abs.(root).^2)
        push!(start_system, build_unitary(scaled_root, start_root, num_vars))
    end

    return start_system, start_root
end

# input a system of polynomials, returns a randomly sampled start system and
# start root
function build_start_system(F, num_vars)
    num_funcs = length(F)
    start_root, null_spaces = sample_linear_intersection(num_funcs, num_vars)
    start_system = []
    for idx in 1:num_funcs
        init_root = sample_zero_set(F[idx], num_vars)
        push!(start_system, map_init_to_start(F[idx], init_root, start_root,
                                              null_spaces[idx], num_funcs,
                                              num_vars))
    end

    return start_system, start_root
end

function build_random_unitary(num_vars)
    svd_res = svd(rand(ComplexF64,num_vars,num_vars))
    return svd_res.U * svd_res.Vt
end

# constructs a unitary matrix that maps moved_root to fixed_root using svd
# (balanced Procrustes problem)
function build_unitary(moved_root, fixed_root, num_vars)
    if isapprox(moved_root, fixed_root, atol=TOL)
        return Matrix(1.0*I,num_vars,num_vars)
    end
    svd_res = svd(fixed_root * moved_root')
    return svd_res.U * svd_res.Vt
end

# FIXME (not sure I understand the intution behind this part)
# sample an intersection of linear forms to build random start_root
function sample_linear_intersection(num_funcs, num_vars)
    null_spaces = []
    union_orthog_comp = zeros(ComplexF64, (num_funcs,num_vars))

    for idx in 1:num_funcs
        linear_form = randn(ComplexF64, (1,num_vars))
        svd_res = svd(linear_form, full=true)
        # for linear forms the rank is automatically one, so the orthogonal
        # complement is spanned by the first row, and the null space is spanned
        # by remaining rows
        union_orthog_comp[idx,:] = svd_res.Vt[1,:]
        push!(null_spaces, svd_res.Vt[2:end,:])
    end

    svd_res = svd(union_orthog_comp, full=true)
    num_rows, _ = size(svd_res.Vt)
    r = length(svd_res.S)
    random_coeffs = randn(ComplexF64, (1,num_rows-r))
    start_root = random_coeffs * svd_res.Vt[(r+1):end,:]

    return reshape(start_root / norm(start_root), num_vars), null_spaces
end

# intersect zero set of inputted polynomial with a random line, sample their
# intersection (this yields a random initial point in the zero set of the given
# polynomial)
function sample_zero_set(f, num_vars)
    P = randn(ComplexF64, num_vars)
    Q = randn(ComplexF64, num_vars)
    y = randn(ComplexF64)

    x, _ = newton!(1.0, x -> f(P*x + Q*y))
    init_root = P*x + Q*y
    # FIXME what if init_root has norm zero?
    return init_root / norm(init_root)
end

# FIXME (not sure I understand the intution behind this part)
# constructs a unitary matrix that maps init_root to start_root (while also
# satisfying a bonus tangent condition)
function map_init_to_start(f, init_root, start_root, null_space, num_funcs, num_vars)
    grad_f = reshape(gradient(X->real(f(X)), init_root)[1] |> conj,
                     (1,num_vars))
    svd_res = svd(grad_f, full=true)
    tangent_space = svd_res.Vt[2:end,:]

    # write init_root in terms of basis for tangent space
    # (tangent space is a matrix with dims (num_vars-1) x num_vars, and init
    # root is a vector with dims num_vars x 1)
    alpha = conj(tangent_space) * init_root
    println(transpose(tangent_space)*alpha - init_root)
    alpha = reshape(init_root,(1,num_vars)) * tangent_space'
    println(alpha*tangent_space-reshape(init_root, (1,num_vars)))
    # @assert isapprox(reshape(tangent_space,(num_vars,num_vars-1))*alpha - init_root, zeros(num_vars,1), atol=TOL)
    @assert false

    # write start_root in terms of basis for null space (computed in
    # sample_linear_intersection)
    # (null space is a matrix with dims (num_vars-1) x num_vars, and start root
    # is a vector with dims num_vars x 1)
    beta = conj(null_space) * start_root
    @assert isapprox(transpose(null_space)*beta - start_root, zeros(num_vars,1), atol=TOL)

    # compute matrix Gamma such that Gamma alpha = beta
    # FIXME check that alpha doesn't need to be conjugated?
    # svd_res = svd(beta * reshape(alpha,(1,num_vars-1)))
    svd_res = svd(beta * alpha')
    Gamma = svd_res.U * svd_res.Vt
    println(Gamma * alpha - beta)
    @assert isapprox(Gamma * alpha - beta, zeros(num_vars-1,1), atol=TOL)

    # FIXME check tangent space doesn't need to be conjugated
    svd_res = svd(reshape(null_space,(num_vars,num_vars-1)) * Gamma *
                  tangent_space)
    res = svd_res.U * svd_res.Vt
    @assert isapprox(res * reshape(tangent_space,(num_vars,num_vars-1)),
                     reshape(null_space,(num_vars,num_vars-1))*Gamma,atol=TOL)
    return svd_res.U * svd_res.Vt
end
