#=

Utility functions for examples

=#

function rand_mat(dim, complex=false)
    if complex
        return rand(Float64,(dim,dim)) + im*rand(Float64,(dim,dim))
    else
        return rand(Float64,(dim,dim))
    end
end

struct WaringPoly
    num_vars::Int
    deg::Int
    rank::Int
    # can we include that M is a num_vars x rank dimensioned array?
    M::Array{ComplexF64}
end

WaringPoly(num_vars,deg,rank) = WaringPoly(num_vars,deg,rank,rand(ComplexF64,num_vars,rank))

function old_evaluate_waring_poly(X, params::WaringPoly)
    return sum([sum(params.M[:,idx] .* X)^params.deg for idx in 1:params.rank])
end

function old_evaluate_waring_system(X, params::Vector{WaringPoly})::Vector{ComplexF64}
    return [sum([sum(p.M[:,idx] .* X)^p.deg for idx in 1:p.rank]) for p in params]
end

function evaluate_waring_poly(X, params::WaringPoly)::ComplexF64
    # FIXME Enzyme doesn't like the intermediate array that is built in the below call
    # I think (error message includes ``Unknown object of type Array{ComplexF64}"), and
    # indeed Julia performance tips suggest not building such intermediate arrays anyway
    # and just summing directly, so here we are.
    # return sum([sum(params.M[:,idx] .* X)^params.deg for idx in 1:params.rank])
    res = 0.0 + 0*im
    for idx in 1:params.rank
        res += sum(params.M[:,idx] .* X)^params.deg
    end
    return res
end

function evaluate_waring_system(X, params::Vector{WaringPoly})::Vector{ComplexF64}
    # return [sum([sum(p.M[:,idx] .* X)^p.deg for idx in 1:p.rank]) for p in params]
    res = zeros(ComplexF64, length(params))
    for i in 1:length(params)
        P = params[i]
        for j in 1:P.rank
            res[i] += sum(P.M[:,j] .* X)^P.deg
        end
    end
    return res
end

function symbolic_evaluate_waring_poly(X, params::WaringPoly)
    return sum([sum(params.M[:,idx] .* X)^params.deg for idx in 1:params.rank])
end

function construct_waring_system_type_safe(rank, degrees, num_vars)::Vector{WaringPoly}
    F = []
    for deg in degrees
        push!(F, WaringPoly(num_vars,deg,rank))
    end
    return F
end

function build_my_system(degrees, num_vars)
    F = []
    for deg in degrees
        M = [rand_mat(deg) for _ in 1:num_vars]
        # affine
        # push!(F, X -> det(sum([X[i]*M[i] for i in 1:(num_vars)]))-1)
        # projective
        push!(F, X -> det(sum([X[i]*M[i] for i in 1:(num_vars-1)]))-1*X[end]^deg)
    end
    return F
end

function build_det_poly_system(degrees, num_vars)
    F = []
    for deg in degrees
        M = [rand_mat(deg) for _ in 1:num_vars]
        push!(F, X -> det(sum([X[i]*M[i] for i in 1:num_vars])))
    end
    return F
end

function build_diff_det_poly_system(degrees, num_vars)
    F = []
    for deg in degrees
        M = [rand_mat(deg) for _ in 1:num_vars]
        N = [rand_mat(deg) for _ in 1:num_vars]
        push!(F, X -> det(sum([X[i]*M[i] for i in 1:num_vars])) -
                      det(sum([X[i]*N[i] for i in 1:num_vars])))
    end
    return F
end

function compare_zero!(roots, new_root, num_vars)
    if isnan(new_root[1])
        if haskey(roots,new_root)
            roots[new_root] += 1
            return
        else
            roots[new_root] = 1
            return
        end
    end

    @assert (!isapprox(new_root, zeros(num_vars,1), atol=TOL))
    for (z, count) in roots
        @assert (!isapprox(z[1], 0.0, atol=TOL))
        ratio = z[1] ./ new_root[1]
        if isapprox(z.-(ratio .* new_root), zeros(num_vars,1), atol=TOL)
            roots[z] += 1
            return
        end
    end
    roots[new_root] = 1
end

function refine(final_root, F)
    refine_count = 0
    prev_root = final_root
    next_root = []
    root_err = 1.0
    while (!isapprox(root_err, 0.0, atol=TOL))
        next_root, _ = newton!(final_root, F)
        root_err = sum(abs.(prev_root - next_root).^2)
        refine_count += 1
    end
    return next_root, refine_count
end
