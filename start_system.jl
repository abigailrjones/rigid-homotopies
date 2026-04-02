include("utils.jl")
using LinearAlgebra: svd, diagm, eigvals, I

# input a vector containing a zero of each polynomial in the system, returns
# the associated start system and start root
function build_start_system(system, init_roots, num_vars)
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
function build_start_system(system, degrees::Vector{Int}, num_vars)
    num_funcs = length(system)
    start_root, null_spaces = sample_linear_intersection(num_funcs, num_vars)
    start_system = []
    for idx in 1:num_funcs
        init_root = sample_zero_set(system[idx], num_vars, degrees[idx])
        check_sampled_init_root(system[idx], init_root)
        push!(start_system, map_init_to_start(system[idx], init_root, start_root,
                                              null_spaces[idx], num_funcs,
                                              num_vars))
    end

    check_build_start_system(system,start_system,start_root,num_funcs)
    return start_system, start_root
end

function build_random_unitary(num_vars)
    svd_res = svd(rand(ComplexF64,num_vars,num_vars))
    return svd_res.U * svd_res.Vt
end

# constructs a unitary matrix that maps moved_root to fixed_root using svd
# (balanced Procrustes problem)
function build_unitary(moved_root, fixed_root, num_vars)
    if isapprox(moved_root, fixed_root, atol=eps(Float64)^0.75)
        return Matrix(1.0*I,num_vars,num_vars)
    end
    svd_res = svd(fixed_root * moved_root')
    return svd_res.U * svd_res.Vt
end

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
    @assert (!isapprox(start_root, zeros(1,num_vars), atol=eps(Float64)^0.75))

    return reshape(start_root / norm(start_root), num_vars), null_spaces
end

# intersect zero set of inputted polynomial with a random line, sample their
# intersection (this yields a random initial point in the zero set of the given
# polynomial)
function sample_zero_set(func, num_vars, deg)
    if num_vars == 1
        throw(ErrorException("Root finding for single variable homogeneous \
                             systems is ill-behaved and not supported."))
    end

    local init_root
    for iter in 1:100
        PQ = randn(ComplexF64, num_vars, 2)
        try
            sol = [1.0 + 0*im, 1.0]
            newton!(sol, [func], [PQ], max_iter=100)
            init_root = PQ * sol
            check_sampled_init_root(func, init_root)
        catch e
            # run companion matrix
            D = deg==1 ? 2 : 2^ceil(Int64, log2(deg))
            coeffs = zeros(ComplexF64,D+1)
            compute_deg_components!(coeffs, x -> func(PQ * [x,1.0+0*im]), 1.0+0*im, D)
            @assert (!isapprox(coeffs[deg+1], 0.0, atol=eps(Float64)^0.75))
            M = diagm(-1 => ones(ComplexF64, deg-1))
            M[1:end, end] = -1*(coeffs[1:deg] / coeffs[deg+1])
            roots = eigvals!(M)
            for root in roots
                init_root = PQ * [root, 1.]
                try
                    newton!(init_root, [func], max_iter=100)
                catch
                    continue
                else
                    if isapprox(func(init_root), 0.0, atol=eps(Float64)^0.75)
                        # make sure root will remain a root after scaling
                        if (log(norm(func(init_root)))-log(eps(Float64)^0.75))/deg < log(norm(init_root))
                            check_sampled_init_root(func, init_root)
                            return init_root / norm(init_root)
                        end
                    end
                end
            end
        else
            return init_root / norm(init_root)
        end
    end
    throw(ErrorException("Failed to sample an initial zero."))
end

function map_init_to_start(func, init_root, start_root, null_space, num_funcs, num_vars)
    grad_f = reshape(build_gradient_reverse!([0.0*im], init_root, func), (1,num_vars))
    svd_res = svd(grad_f, full=true)
    # getting columns of V as rows by just taking complex conjugate of Vt
    tangent_space = conj(svd_res.Vt[2:end,:])

    # write init_root in terms of basis for tangent space
    # (tangent space is a matrix with dims (num_vars-1) x num_vars, and init
    # root is a vector with dims num_vars x 1)
    alpha = conj(tangent_space) * init_root
    @assert isapprox(transpose(tangent_space)*alpha - init_root,
                     zeros(num_vars,1), atol=eps(Float64)^0.75)

    # write start_root in terms of basis for null space (computed in
    # sample_linear_intersection)
    # (null space is a matrix with dims (num_vars-1) x num_vars, and start root
    # is a vector with dims num_vars x 1)
    beta = conj(null_space) * start_root
    @assert isapprox(transpose(null_space)*beta - start_root,
                     zeros(num_vars,1), atol=eps(Float64)^0.75)

    # compute matrix Gamma such that Gamma alpha = beta
    svd_res = svd(beta * alpha')
    Gamma = svd_res.U * svd_res.Vt
    @assert isapprox(Gamma * alpha - beta, zeros(num_vars-1,1), atol=eps(Float64)^0.75)

    svd_res = svd(transpose(null_space) * Gamma * conj(tangent_space))
    res = svd_res.U * svd_res.Vt
    @assert isapprox(res*transpose(tangent_space) -
                     transpose(null_space)*Gamma, zeros(num_vars,num_vars-1),
                     atol=eps(Float64)^0.75)

    @assert isapprox(res * init_root - start_root, zeros(num_vars,1), atol=eps(Float64)^0.75)
    return res
end
