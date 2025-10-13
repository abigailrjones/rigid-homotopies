include("../rigid_hom.jl")

f = X -> X[1]^5 - 2*X[3]*X[1]^4 + X[2]^5
g = X -> X[1]^2 + X[2]^2 - X[3]^2
F = [f,g]

num_funcs = 2
num_vars = 3
max_degree = 5
max_iter = 10000
use_heuristic = true

init_roots = [[1,1,1],[1,0,1]]

# solve(F, num_funcs, num_vars, max_degree, max_iter, use_heuristic, init_roots)
solve(F, num_funcs, num_vars, max_degree, max_iter, use_heuristic)

println()
