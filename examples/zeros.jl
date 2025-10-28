include("../rigid_hom.jl")
include("example_utils.jl")

TOL = eps()*10000

num_vars = 4 # free
degrees = [4,4,4] # had issues when deg >= num_funcs for any degree
num_funcs = length(degrees)
max_degree = maximum(degrees)
max_iter = 1000
use_heuristic = true
mid_print = false

F = build_diff_det_poly_system(degrees, num_vars)

roots = Dict()
for idx in 1:10
    final_root, _ = solve(F, num_funcs, num_vars, max_degree, max_iter;
                          use_heuristic=use_heuristic, mid_print=mid_print)
    compare_zero!(roots, final_root, num_vars)
end

println(length(roots))

for (z,count) in roots
    println("Zero: $z, count: $count")
end
