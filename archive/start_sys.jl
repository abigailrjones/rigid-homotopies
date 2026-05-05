include("../start_system.jl")
# using Plots

num_vars = 5
deg = 10
rank = 6
P = WaringPoly(num_vars,deg,rank)

# F = X -> 5*X[1]^3 + 2*X[1]^2 + 3*X[1] + 4

sample_zero_set(P,num_vars,deg);
