using BenchmarkTools
using Zygote

include("../example_utils.jl")


function direct(F, input1, input2)
    Zygote.jacobian(X -> real(FF(X)), input1)[1] |> conj
    Zygote.jacobian(X -> real(FF(X)), input2)[1] |> conj
    return
end

function symbolic(F, input1, input2)
    jac = Y -> Zygote.jacobian(X -> real(FF(X)), Y)[1] |> conj
    jac(input1)
    jac(input2)
    return
end

a = 5
b = 10

D = 2 # degree
r = 3 # waring rank (r > D)
base = 2 # the factor we increase number of variables by

# open("examples/profiling/data/data_check_jac_complexity.txt", "w") do f
    # empty file of previous contents
# end

# checking complexity of single polynomial
for i in a:b
    DD = ones(Int64, base^i) * D

    # note that we use ``global" because btime only works with global variables
    global F = build_waring_system(r, DD, base^i)
    global function ff(X::Vector) return F[1](X) end
    global function FF(X::Vector) return [F[idx](X) for idx in 1:length(F)] end
    global input1 = rand(ComplexF64, base^i)
    global input2 = rand(ComplexF64, base^i)

    num_res = @btimed direct(FF, input1, input2)
    sym_res = @btimed symbolic(FF, input1, input2)
    println(i, " ", num_res.time, " ", sym_res.time)
    #=
    open("examples/profiling/data/data_check_jac_complexity.txt", "a") do file
        write(file, "$i $(poly_res.time) $(sys_res.time)\n")
    end
    =#
end
