using BenchmarkTools
using Zygote
using Plots

include("../example_utils.jl")


a = 11
b = 15

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
    global input = rand(ComplexF64, base^i)

    poly_res = @btimed Zygote.gradient(X -> real(ff(X)), input)[1] |> conj
    sys_res = @btimed Zygote.jacobian(X -> real(FF(X)), input)[1] |> conj
    open("examples/profiling/data/data_check_jac_complexity.txt", "a") do file
        write(file, "$i $(poly_res.time) $(sys_res.time)\n")
    end
end
