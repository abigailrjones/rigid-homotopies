#=

The starting polynomial system is a standard eigenvalue problem: F(v,lambda) =
A v - lambda v.

The target polynomial system is like a twisted eigenvalue problem. It may be an
eigenvalue problem itself, but I struggled to put it in that exact form, hence
the ``twisted" designation.

The primary goal of this example is to simply track a path from a known start
point to a random terminal point. We can then check whether path jumping is
occurring.

=#

include("../rigid_hom.jl")

mat_dims = 2
num_vars = mat_dims + 1
M = rand(Float64,(mat_dims,mat_dims)) + im*rand(Float64,(mat_dims,mat_dims))
# make A a skew Hermitian matrix (then all eigenvalues are complex, and if A
# is invertible, it will follow that log(A) is defined)
# (since M is random, it is probably true that A is invertible)
A = M - M'

# we are computing eivalue/eigenvector pairs, where X = [v, lambda]
# the start_system (and resulting path) are chosen so that lambda and v do not
# get mixed together as we move along the path
F = []
for idx in 1:mat_dims
    push!(F, X -> sum(A[idx,:] .* X[1:end-1]) - X[end]*X[idx])
end
# add a final constraint that forces eigenvectors away from the zero vector
push!(F, X -> sum(abs2.(X[1:end-1])) - 1)
num_funcs = length(F)

max_degree = 2
max_iter = 10000
use_heuristic = true

start_system = [Matrix{Float64}(I,num_vars,num_vars) for _ in 1:num_funcs]
eivals, eivecs = eigen(A)
V = eivecs[:,1]
start_root = push!(V, eivals[1])

target_system = []
for idx in 1:num_funcs
    mat = rand(Float64,(num_vars,num_vars)) + im*rand(Float64,(num_vars,num_vars))
    mat[:,end] = zeros(num_vars)
    mat[end,:] = zeros(num_vars)
    mat[end, end] = 1
    push!(target_system, mat)
end

# at t=1, path is identity (start system)
# at t=0, path is target system (determined by the randomly generated matrices above)
path = t -> [exp((1-t) * log(mat)) for mat in target_system]

final_root, _ = solve(F, num_funcs, num_vars, max_degree, max_iter, use_heuristic,
                      start_system, start_root, path)
# this last line is just a janky way of forcing julia to not print the return
# statement of the previous line, which just clutters up the terminal window
println()
