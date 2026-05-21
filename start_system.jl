include("utils.jl")
using LinearAlgebra: svd, diagm, eigvals!, I, det

# input a vector containing a zero of each polynomial in the system, returns
# the associated start system and start root
function build_start_system(system, init_roots, num_vars)
    V = build_random_unitary(eltype(init_roots[1]), num_vars)
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
function build_start_system(::Type{T}, system, degrees::Vector{Int}, num_vars) where T <: Union{ComplexF64, Float64}
    num_funcs = length(system)
    start_root, null_spaces = sample_linear_intersection(T, num_funcs, num_vars)
    start_system = Vector{Matrix{T}}(undef, num_funcs)
    for idx in 1:num_funcs
        init_root = sample_zero_set(T, system[idx], num_vars, degrees[idx])
        start_system[idx] = map_init_to_start(system[idx], init_root,
                                              start_root, null_spaces[idx],
                                              num_vars)
    end

    check_build_start_system(system,start_system,start_root,num_funcs)
    return start_system, start_root
end

function build_random_unitary(::Type{T}, num_vars) where T <: Union{ComplexF64, Float64}
    svd_res = svd(randn(T,num_vars,num_vars))
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

function sample_linear_intersection(::Type{T}, num_funcs, num_vars) where T <: Union{ComplexF64, Float64}
    null_spaces = Vector{Matrix{T}}(undef, num_funcs)
    union_orthog_comp = zeros(T, (num_funcs,num_vars))

    for idx in 1:num_funcs
        linear_form = randn(T, (1,num_vars))
        svd_res = svd(linear_form, full=true)
        # for linear forms the rank is automatically one, so the orthogonal
        # complement is spanned by the first row, and the null space is spanned
        # by remaining rows
        union_orthog_comp[idx,:] = svd_res.Vt[1,:]
        null_spaces[idx] = conj(svd_res.Vt[2:end,:])
    end

    svd_res = svd(union_orthog_comp, full=true)
    num_rows, _ = size(svd_res.Vt)
    r = length(svd_res.S)
    random_coeffs = randn(T, (1,num_rows-r))
    start_root = random_coeffs * conj(svd_res.Vt[(r+1):end,:])
    @assert (!isapprox(start_root, zeros(1,num_vars), atol=eps(Float64)^0.75))

    return reshape(start_root / norm(start_root), num_vars), null_spaces
end

# intersect zero set of inputted polynomial with a random line, sample their
# intersection (this yields a random initial point in the zero set of the given
# polynomial)
function sample_zero_set(::Type{T}, func, num_vars, deg) where T <: Union{ComplexF64, Float64}
    if num_vars == 1
        throw(ErrorException("Root finding for single variable homogeneous \
                             systems is ill-behaved and not supported."))
    end

    local init_root = zeros(T, num_vars)
    for iter in 1:100
        PQ = randn(T, num_vars, 2)
        try
            sol = ones(T, 2)
            newton!(sol, [func], [PQ], max_iter=100)
            init_root .= PQ * sol
            check_sampled_init_root(func, init_root)
        catch e
            continue
        else
            @assert eltype(init_root) == T
            return init_root / norm(init_root)
        end
    end
    # run companion matrix
    println("Trying companion matrix.")
    # TODO is it worth running companion matrix multiple times (for different PQ)?
    for iter in 1:1
        PQ = randn(T, num_vars, 2)
        D = deg==1 ? 2 : 2^ceil(Int64, log2(deg))
        coeffs = zeros(ComplexF64,D+1)
        compute_deg_components!(coeffs, x -> func(PQ * [x,one(T)]), one(T), D)
        @assert (!isapprox(coeffs[deg+1], 0.0, atol=eps(Float64)^0.75))
        M = diagm(-1 => ones(T, deg-1))
        M[1:end, end] = -1*(coeffs[1:deg] / coeffs[deg+1])
        roots = eigvals!(M)
        aux_root = zeros(ComplexF64, num_vars)
        for root in roots
            if T == Float64
                aux_root .= PQ * [root, 1.0+0*im]
                # if possible root doesn't have the right type, move on
                if !isapprox(imag(aux_root), zeros(num_vars), atol=eps(Float64)^0.75)
                    continue
                end
                init_root .= real(aux_root)
            else # type is complex and we don't need to worry about the type of root
                init_root .= PQ * [root, 1.0+0*im]
            end
            try
                newton!(init_root, [func], max_iter=100)
            catch
                continue
            else
                if isapprox(func(init_root), 0.0, atol=eps(Float64)^0.75)
                    # make sure root will remain a root after scaling
                    if (log(norm(func(init_root)))-log(eps(Float64)^0.75))/deg < log(norm(init_root))
                        check_sampled_init_root(func, init_root)
                        @assert eltype(init_root) == T
                        return init_root / norm(init_root)
                    end
                end
            end
        end
    end
    throw(ErrorException("Failed to sample an initial zero."))
end

function map_init_to_start(func, init_root::Vector{T}, start_root::Vector{T}, null_space::Array{T}, num_vars) where T <: Union{ComplexF64, Float64}
    grad_f = reshape(build_gradient_reverse!(zeros(T,1), init_root, func), (1,num_vars))
    svd_res = svd(grad_f, full=true)
    # getting columns of V as rows by just taking complex conjugate of Vt
    # (tangent space is orthgonal to gradient vector, hence null space)
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
    if T == Float64
        if !isapprox(det(res), 1.0, atol=eps(Float64)^0.75)
            svd_res.Vt[end,:] .*= -1
            res = svd_res.U * svd_res.Vt
            @assert isapprox(det(res), 1.0, atol=eps(Float64)^0.75)
        end
    end
    @assert isapprox(res*transpose(tangent_space) -
                     transpose(null_space)*Gamma, zeros(num_vars,num_vars-1),
                     atol=eps(Float64)^0.75)

    @assert isapprox(res * init_root - start_root, zeros(num_vars,1), atol=eps(Float64)^0.75)
    # if T == Float64, this assert checks that the path we define later will stay in the reals
    @assert eltype(log(res)) == T
    return res
end
