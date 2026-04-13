include("utils.jl")
using LinearAlgebra: svdvals

# Note to self: verified real version via plotting
function sample_unit_ball(dim::Integer, num_pts::Integer)
    x = randn(ComplexF64,(num_pts,dim))
    r = rand(Float64,(num_pts,)).^(1/dim)
    norm = sqrt.(sum(abs2, x; dims=2))
    return [x[i,:]*r[i]/norm[i] for i in 1:num_pts]
end

function estimate_gammaprob(func::Function,grad_at_input,eta,DD::Integer,num_vars)
    D = DD==1 ? 2 : 2^ceil(Int64, log2(DD))
    d0h_sq_norm = 0.0
    for elt in grad_at_input
        d0h_sq_norm += abs(elt)^2
    end
    s = ceil(Int64, 1 + log(2, D/eta))
    rand_w = sample_unit_ball(num_vars, s)
    sum_squared_components = zeros(D+1)
    deg_components = zeros(ComplexF64,D+1)
    for w in rand_w
        # compute each degree component of h evaluated at w, square elementwise
        # and add to sum
        compute_deg_components!(deg_components,func,w,D)
        sum_squared_components .+= abs.(deg_components).^2
    end
    return_est = 0
    fac = k -> binomial(num_vars+k, k)/d0h_sq_norm/s
    for k in 2:D
        new_est = (32*(num_vars-1)*k)^(1/(2-2/k))*(fac(k)*sum_squared_components[k+1])^(1/(2*k-2))
        if new_est > return_est
            return_est = new_est
        end
    end
    return return_est
end

# condition number is denoted by \kappa in the papers
# in particular, we consider the L2 norm of the matrix L(F_t, z) (defined in
# (15) in RH1), which is the inverse of the smallest singular value of L
function compute_condition_num(jac)
    if length(size(jac))==1
        # jacobian is one-dimensional, so row_norms is a scalar
        row_norms = 1.0 / sqrt(sum(abs2, jac))
    else
        row_norms = 1.0 ./ sqrt.(sum(abs2, jac; dims=2))
    end
    L = jac .* row_norms
    # note that svdvals lists singular values in descending order
    return 1.0 / svdvals(L)[end]
end

function choose_timestep!(system, W_t, input, D, max_iter, num_funcs, num_vars, iter, prog_data; epsilon=1e-8)
    sum_g_sq = 0.0
    jac, _ = build_jacobian_reverse(input, system, W_t)
    for idx in 1:num_funcs
        func = X -> system[idx](W_t[idx] * (X + input))
        grad_at_input = jac[idx,:]
        # sum_g_sq += estimate_gammaprob(func,grad_at_input,epsilon/((num_vars-1)*max_iter),D,num_vars)^2
        sum_g_sq += estimate_gammaprob(func,grad_at_input,epsilon,D,num_vars)^2
    end
    cond_num = compute_condition_num(jac)^2
    sqrt_sum_g_sq = sqrt(sum_g_sq)
    if (sqrt_sum_g_sq < prog_data[6]) prog_data[6] = sqrt_sum_g_sq end
    if (sqrt_sum_g_sq > prog_data[7]) prog_data[7] = sqrt_sum_g_sq end
    prog_data[8] = prog_data[8] + (sqrt_sum_g_sq - prog_data[8])/iter
    if (cond_num < prog_data[9]) prog_data[9] = cond_num end
    if (cond_num > prog_data[10]) prog_data[10] = cond_num end
    prog_data[11] = prog_data[11] + (cond_num - prog_data[11])/iter

    return 1/(240 * cond_num^2 * sqrt_sum_g_sq)
end
