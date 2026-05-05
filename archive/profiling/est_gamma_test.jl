include("../../utils.jl")
include("../../choose_timestep.jl")
# using ProfileView

function run_est_gamma(system, input, D, max_iter, num_funcs, num_vars; eps=1e-8)
    jac, _ = build_jacobian_reverse(input, system)
    @profview for idx in 1:num_funcs estimate_gammaprob(X -> system[idx](X + input),jac[idx,:],eps/((num_vars-1)*max_iter),D,num_vars)^2 end
    # for idx in 1:num_funcs @time (estimate_gammaprob(X -> system[idx](X + input),jac[idx,:],eps/((num_vars-1)*max_iter),D,num_vars)^2) end
end

num_vars = 3
num_funcs = 100_000
degs = ones(Int,num_funcs)*4
ranks = ones(Int,num_funcs)*3
system = build_waring_system(num_vars, degs, ranks)
input = rand(ComplexF64, num_vars)

#=
num_vars = 3
degs = [3]
ranks =[3]
M = [1.0 + 0*im 2 3; 4 2.1 2; 0.1 0 2.3]
system = [WaringPoly(num_vars,degs[1],ranks[1],M)]
num_funcs = length(system)
input = [1.0 + 0*im, 0.1, 3]
=#

run_est_gamma(system, input, maximum(degs), 1_000_000_000, num_funcs, num_vars)
