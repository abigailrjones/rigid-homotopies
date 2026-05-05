include("../rigid_hom.jl")
include("example_utils.jl")
include("../utils.jl")

import HomotopyContinuation

TOL = eps()*10000

open("examples/data/data_finite_prec.txt", "w") do f
    # empty file of previous contents
end

num_vars = 4 # free
degrees = [4,4,4] # had issues when deg >= num_funcs for any degree
num_funcs = length(degrees)
max_degree = maximum(degrees)
max_iter = 1000
use_heuristic = true
mid_print = false

# F = build_my_system(degrees, num_vars)
F = build_diff_det_poly_system(degrees, num_vars)
# F = []
# push!(F, X -> det(X[1]*[1 2; 3 4] + X[2]*[1 -1; 0 1]))
# push!(F, X -> det(X[1]*[3 2; -1 3] + X[2]*[1 4; -1 1]))

# build start system and path here, so that randomness is fixed throughout the example
#=
start_system, start_root = build_start_system(F, num_vars)
check_build_start_system(F, start_system, start_root, num_funcs)
path = build_path(start_system)
check_build_path(path, start_system, num_vars)
=#

roots = Dict()
roots[NaN] = 0
for idx in 1:2000
    final_root, _ = solve(F, num_funcs, num_vars, max_degree, max_iter; use_heuristic=use_heuristic, mid_print=mid_print)
    compare_zero!(roots, final_root, num_vars)
end

println(length(roots))

#=
for (root, count) in roots
    open("examples/data_finite_prec.txt", "a") do f
        write(f, "$root $count\n")
    end
end
=#

HomotopyContinuation.ModelKit.@var x[1:num_vars]
FF = [F[idx](x) for idx in 1:num_funcs]
res = HomotopyContinuation.solve(FF)#, homvar=x[end])
# affine
# solutions = [(path.solution, path.accuracy) for path in res]
# projective
solutions = [(path.solution ./ path.solution[end], path.accuracy) for path in res]
count = []
for (sol,acc) in solutions
    if isapprox(acc,0.0,atol=TOL)
        # println(sol)
        push!(count, 0)
    end
end
println(length(count))

open("examples/data/data_finite_prec.txt", "a") do f
    write(f, "$initial_dt $final_root\n")
end
