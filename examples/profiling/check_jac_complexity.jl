using BenchmarkTools
using LinearAlgebra: I
import Zygote
using Enzyme

include("../example_utils.jl")

@assert false

function build_gradient_forward(guess, P)
    return Enzyme.autodiff(Forward, evaluate_waring_poly, Duplicated,
                           BatchDuplicated(guess, d_guess), Const(P))
end

function build_jacobian_forward(guess, params)
    return Enzyme.autodiff(Forward, evaluate_waring_system, Duplicated,
                           BatchDuplicated(guess, d_guess), Const(params))
end

function build_gradient_reverse(guess, P)
    res = zeros(ComplexF64, P.num_vars)
    Enzyme.autodiff(ReverseHolomorphic, evaluate_waring_poly, Active, Duplicated(guess, res), Const(P))
    return res
end

function build_jacobian_reverse(guess, params)
    # TODO can we just have one allocation for jac the whole time and pass it in?
    jac = zeros(ComplexF64, length(params), params[1].num_vars)
    for idx in 1:length(params)
        view(jac,idx,:) .= build_gradient_reverse(guess, params[idx])
    end
    return jac
end

a = 1
b = 20

deg = 2 # degree
rank = 3 # waring rank (r > D)
base = 2 # the factor we increase number of variables by

open("examples/profiling/data/data_check_jac_complexity.txt", "w") do f
    # empty file of previous contents
end

# checking complexity of single polynomial
for i in a:b
    num_vars = base^i
    num_funcs = num_vars - 1

    global id = (1.0+0*im)*Matrix(I, num_vars, num_vars)
    global d_guess = Tuple(id[:,idx] for idx in 1:num_vars)

    # note that we use ``global" because btime only works with global variables
    global Polys = Vector{WaringPoly}(undef, num_funcs)
    for idx in 1:num_funcs
        Polys[idx] = WaringPoly(num_vars, deg, rank)
    end
    global input = rand(ComplexF64, base^i)

    # ENZYME (forward)
    poly_res_f = @btimed build_gradient_forward(input, Polys[1])
    sys_res_f = @btimed build_jacobian_forward(input, Polys)

    # ENZYME (reverse)
    poly_res_r = @btimed build_gradient_reverse(input, Polys[1])
    sys_res_r = @btimed build_jacobian_reverse(input, Polys)

    # ZYGOTE
    global f_t = X -> old_evaluate_waring_poly(X, Polys[1])
    old_poly_res = @btimed Zygote.gradient(X -> real(f_t(X)), input)[1] |> conj
    global F_t = X -> old_evaluate_waring_system(X, Polys)
    old_sys_res = @btimed Zygote.jacobian(X -> real(F_t(X)), input)[1] |> conj

    open("examples/profiling/data/data_check_jac_complexity.txt", "a") do file
        write(file, "$(base^i) $(poly_res_f.time) $(sys_res_f.time) $(poly_res_r.time) $(sys_res_r.time) $(old_poly_res.time) $(old_sys_res.time)\n")
    end
end
