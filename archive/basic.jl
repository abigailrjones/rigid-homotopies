include("../rigid_hom.jl")

system = [X -> sum(X), X -> X[1]*X[2]]
solve(system, 2, 3, [1,2], 100000, mid_print=true);
