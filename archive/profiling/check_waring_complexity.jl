using BenchmarkTools

include("../example_utils.jl")

# I keep accidentally overwriting files that took a long time to make, so this
# assert will stop me (I must be stopped)
@assert false

a = 1
b = 20

deg = 2 # degree
rank = 3 # waring rank (rank > D)
base = 2 # the factor we increase number of variables by

open("examples/profiling/data/data_check_waring_complexity.txt", "w") do f
    # empty file of previous contents
end

# checking complexity of evaluating a single polynomial and a polynomial system
for i in a:b
    local num_vars = base^i
    local num_funcs = num_vars - 1

    # note that we use ``global" because btime only works with global variables
    global Polys = Vector{WaringPoly}(undef, num_funcs)
    for idx in 1:num_funcs
        Polys[idx] = WaringPoly(num_vars, deg, rank)
    end
    global input = rand(ComplexF64, num_vars)

    poly_res = @btimed evaluate_waring_poly(input, Polys[1])
    sys_res = @btimed evaluate_waring_system(input, Polys)
    open("examples/profiling/data/data_check_waring_complexity.txt", "a") do file
        write(file, "$(base^i) $(poly_res.time) $(sys_res.time) $(old_poly_res.time) $(old_sys_res.time)\n")
    end
end
