using BenchmarkTools: @btimed
using LinearAlgebra: I
using Enzyme

include("../example_utils.jl")


function build_jacobian(guess, params)
    num_vars = params[1].num_vars
    id = (1.0+0*im)*Matrix(I, num_vars, num_vars)
    d_guess = Tuple(id[:,idx] for idx in 1:num_vars)
    return Enzyme.autodiff(Forward, evaluate_waring, Duplicated,
                           BatchDuplicated(guess, d_guess), Const([params[1]]))
end

a = 1
b = 3

D = 2 # degree
r = 3 # waring rank (r > D)
base = 2 # the factor we increase number of variables by

# open("examples/profiling/data/data_check_jac_complexity.txt", "w") do f
    # empty file of previous contents
# end

for i in a:b
    DD = ones(Int64, base^i) * D

    # note that we use ``global" because btime only works with global variables
    global params = construct_waring_system_type_safe(r, DD, base^i)
    global input = rand(ComplexF64, base^i)

    enz_res = @btimed build_jacobian(input, params)
    println(i, " ", enz_res.time)
    #=
    open("examples/profiling/data/data_check_jac_complexity.txt", "a") do file
        write(file, "$i $(poly_res.time) $(sys_res.time)\n")
    end
    =#
end
