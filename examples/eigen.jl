#=

The starting polynomial system is a standard eigenvalue problem: F(v,lambda) =
A v - lambda v.

The target polynomial system is another eigenvalue problem, but for the matrix
whose ith row is the ith row of A times (on the right) the conjugate transpose
of the ith matrix in the target system. The target system is chosen randomly.

=#

include("../rigid_hom.jl")

num_funcs = 2
num_vars = num_funcs + 1
M = rand(Float64,(num_funcs,num_funcs)) + im*rand(Float64,(num_funcs,num_funcs))
# make A a skew Hermitian matrix (then all eigenvalues are complex, and if A
# is invertible, it will follow that log(A) is defined)
# (since M is random, it is probably true that A is invertible)
A = M - M'

# we are computing eivalue/eigenvector pairs, where X = [v, lambda]
# the start_system (and resulting path) are chosen so that lambda and v do not
# get mixed together as we move along the path
F = [X -> sum(A[idx,:] .* X[1:end-1]) - X[end]*X[idx] for idx in 1:num_funcs]

max_degree = 2
max_iter = 10000
use_heuristic = true

start_system = [Matrix{Float64}(I,num_vars,num_vars) for _ in 1:num_funcs]
eivals, eivecs = eigen(A)
V = eivecs[:,1]
start_root = push!(V, eivals[1])

target_system = []
for idx in 1:num_funcs
    mat = rand(Float64,(num_funcs+1,num_funcs+1)) + im*rand(Float64,(num_funcs+1,num_funcs+1))
    mat[:,end] = zeros(num_funcs+1)
    mat[end,:] = zeros(num_funcs+1)
    mat[end, end] = 1
    push!(target_system, mat)
end

# at t=1, path is identity (start system)
# at t=0, path is target system (determined by the randomly generated matrices above)
path = t -> [exp((1-t) * log(mat)) for mat in target_system]

final_root, _ = solve(F, num_funcs, num_vars, max_degree, max_iter, use_heuristic,
                      start_system, start_root, path)

#=
# TODO check that we understand what this whole problem actually is (that it's
# just an eigenvalue problem after all, and that we almost certainly
# overcomplicated things)
# FIXME I think the program is actually not computing this eigenvalue yet
# (because of something in track_path) so this is a good check to see if I'm
# right
B = Matrix([transpose(A[idx,:]) * target_system[idx][1:end-1,1:end-1] for idx in 1:num_funcs])
println(B)
println(B*final_root[1:end-1] - final_root[end]*final_root[1:end-1])
=#
