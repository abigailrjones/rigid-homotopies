#=

Compute the average (proven) timestep needed over the first N steps for a
set of increasing dimension determinantal problems.

=#

using Plots
plotlyjs()

include("../rigid_hom.jl")
include("det_poly.jl")


num_vars = 10 # free
degrees = [3,3,3,4] # had issues when deg >= num_funcs for any degree
num_funcs = length(degrees)
max_degree = maximum(degrees)
max_iter = 1000
use_heuristic = false
mid_print = false

F = build_det_poly_system(degrees, num_vars)

final_root, _ = solve(F, num_funcs, num_vars, max_degree, max_iter, use_heuristic, mid_print)
