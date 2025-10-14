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
        # orthogonal complement to null space is spanned by first row of Vt (in
        # this case), since null space is spanned by complex conjugate of
        # remaining rows of Vt
        union_orthog_comp[idx,:] = svd_res.Vt[1,:]
        push!(null_spaces, conj(svd_res.Vt[2:end,:]))
    end

    svd_res = svd(union_orthog_comp, full=true)
    num_rows, _ = size(svd_res.Vt)
    r = length(svd_res.S)
    random_coeffs = randn(ComplexF64, (1,num_rows-r))
    start_root = random_coeffs * conj(svd_res.Vt[(r+1):end,:])
    # FIXME FIXME FIXME
    # println("orthog comp:", union_orthog_comp)
    # println("SVD:", svd_res)
    # println("r:",r)
    # println("Vt:",svd_res.Vt[(r+1):end,:])
    # println("Start root:", start_root)
    @assert (!isapprox(start_root, zeros(1,num_vars), atol=TOL))

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
    # println("init_root:",init_root)
    return init_root / norm(init_root)
end

# FIXME (not sure I understand the intution behind this part)
# constructs a unitary matrix that maps init_root to start_root (while also
# satisfying a bonus tangent condition)
function map_init_to_start(f, init_root, start_root, null_space, num_funcs, num_vars)
    grad_f = reshape(gradient(X->real(f(X)), init_root)[1] |> conj,
                     (1,num_vars))
    svd_res = svd(grad_f, full=true)
    # getting columns of V as rows by just taking complex conjugate of Vt
    tangent_space = conj(svd_res.Vt[2:end,:])

    # write init_root in terms of basis for tangent space
    # (tangent space is a matrix with dims (num_vars-1) x num_vars, and init
    # root is a vector with dims num_vars x 1)
    alpha = conj(tangent_space) * init_root
    if (!isapprox(transpose(tangent_space)*alpha - init_root,
                  zeros(num_vars,1), atol=TOL))
        # FIXME FIXME
        #=
        println()
        println("***Failed assert 1:",transpose(tangent_space)*alpha - init_root)
        =#
    end
    # @assert isapprox(transpose(tangent_space)*alpha - init_root,
    #                  zeros(num_vars,1), atol=TOL)

    # write start_root in terms of basis for null space (computed in
    # sample_linear_intersection)
    # (null space is a matrix with dims (num_vars-1) x num_vars, and start root
    # is a vector with dims num_vars x 1)
    beta = conj(null_space) * start_root
    if (!isapprox(transpose(null_space)*beta - start_root, zeros(num_vars,1),
                  atol=TOL))
        # FIXME FIXME
        #=
        println()
        println("***Failed assert 2:",transpose(null_space)*beta - start_root)
        =#
    end
    #@assert isapprox(transpose(null_space)*beta - start_root,
    #                 zeros(num_vars,1), atol=TOL)

    # compute matrix Gamma such that Gamma alpha = beta
    svd_res = svd(beta * alpha')
    Gamma = svd_res.U * svd_res.Vt
    @assert isapprox(Gamma * alpha - beta, zeros(num_vars-1,1), atol=TOL)

    svd_res = svd(transpose(null_space) * Gamma * conj(tangent_space))
    res = svd_res.U * svd_res.Vt
    @assert isapprox(res*transpose(tangent_space) -
                     transpose(null_space)*Gamma, zeros(num_vars,num_vars-1),
                     atol=TOL)

    # @assert isapprox(res * init_root - start_root, zeros(num_vars,1), atol=TOL)
    if (!isapprox(res * init_root - start_root, zeros(num_vars,1), atol=TOL))
        # FIXME FIXME
        #=
        println()
        println("***Failed assert 3:", res * init_root - start_root)
        =#
        # @assert false
    end
    return res
end
