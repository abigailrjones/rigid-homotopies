using BenchmarkTools
include("choose_timestep.jl")

# function old_estimate_gammaprob(system, W_t, input, num_vars, num_funcs, eta,
        # d0h_sq_norm, DD::Integer, rand_w)

num_vars = 20
num_funcs = 19
degrees = [2,3,4,5,6,7,8,9,10,2,3,4,5,6,7,8,9,2,3]
ranks = degrees .+ 1

system = build_waring_system(num_vars, degrees, ranks)
W_t = [rand(ComplexF64, num_vars, num_vars) for idx in 1:num_funcs]
DD = maximum(degrees)
input = rand(ComplexF64, num_vars)
eta = 1e-8

jac, _ = build_jacobian_reverse(input, system, W_t)
d0h_sq_norm = sum(abs.(jac).^2, dims=2)

D = DD==1 ? 2 : 2^ceil(Int64, log2(DD))
s = ceil(Int64, 1 + log(2, D/eta))
rand_w = sample_unit_ball(num_vars, s)

println(@btime estimate_gammaprob(system, W_t, input, num_vars, num_funcs, eta, d0h_sq_norm, DD, rand_w))
println(@btime old_estimate_gammaprob(system, W_t, input, num_vars, num_funcs, eta, d0h_sq_norm, DD, rand_w))
