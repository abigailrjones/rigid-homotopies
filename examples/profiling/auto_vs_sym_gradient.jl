using BenchmarkTools: @btimed
import Zygote
using LinearAlgebra: I
using Enzyme
import SymEngine


include("../example_utils.jl")

# TODO
# - start with small example (performance on this will not be enough on its own
# to make a choice, but we'll be able to more easily set up the framework on
# the small example)
# - then extend to larger examples, where the methods will likely diverge in
# terms of performance

num_vars = 2
deg = 2
rank = 2
# 2 x 2 matrix (num_vars x rank)
M = [1. + 0*im 2.; 3. 4.]

# P(x,y) = 5x^2 + 22xy + 25y^2
P = WaringPoly(num_vars,deg,rank,M)
F_t = X -> evaluate_waring_poly(X, P)

guess = [1. + 0*im, 2.]

# true gradient
# G(x,y) = [10*x + 22*y, 22*x + 50*y]
# println("True solution: $(G(guess[1],guess[2]))\n")

# ZYGOTE
println("Zygote:")
zyg_res = @btimed jac_zygote = Zygote.jacobian(x -> real(F_t(x)), guess)[1] |> conj
println("value: $(zyg_res.value)")
println("time: $(zyg_res.time)")
println("alloc: $(zyg_res.alloc)")
println()

# ENZYME
function build_jacobian(guess, params)
    # TODO update evaluate_waring_system in the style of current
    # evaluate_waring_poly so that Enzyme doesn't have a panic attack
    num_vars = params[1].num_vars
    id = (1.0+0*im)*Matrix(I, num_vars, num_vars)
    d_guess = Tuple(id[:,idx] for idx in 1:num_vars)
    return Enzyme.autodiff(Forward, evaluate_waring_system, Duplicated,
                           BatchDuplicated(guess, d_guess), Const([params[1]]))
end

function build_gradient(guess, P)
    id = (1.0+0*im)*Matrix(I, P.num_vars, P.num_vars)
    d_guess = Tuple(id[:,idx] for idx in 1:P.num_vars)
    return Enzyme.autodiff(Forward, evaluate_waring_poly, Duplicated,
                           BatchDuplicated(guess, d_guess), Const(P))
end

println("Enzyme:")
enzyme_res = @btimed jac_enzyme = build_gradient(guess, P)
println("value: $(enzyme_res.value)")
println("time: $(enzyme_res.time)")
println("alloc: $(enzyme_res.alloc)")
println()

# SYMENGINE
function symbolic_jac(guess, P)
    XX = [SymEngine.symbols("xx_$i") for i in 1:P.num_vars]
    #=
    # FIXME using diff! only saves a few allocations (<10)
    jac_sym = zeros(SymEngine.Basic, num_vars)
    for idx in 1:num_vars
        SymEngine.diff!(jac_sym[idx], symbolic_evaluate_waring_poly(XX, P), XX[idx])
    end
    =#
    jac_sym = [SymEngine.diff(symbolic_evaluate_waring_poly(XX, P), v) for v in XX]
    sub_guess = [XX[idx]=>guess[idx] for idx in 1:num_vars]
    for sub in sub_guess
        jac_sym .= [SymEngine.subs(e, sub) for e in jac_sym]
    end
    return jac_sym
end
println("Symbolic:")
sym_res = @btimed jac_sym = symbolic_jac(guess, P)
println("value: $(sym_res.value)")
println("time: $(sym_res.time)")
println("alloc: $(sym_res.alloc)")
println()

# HOMOTOPYCONTINUATION.jl
# Identical to Symbolic case
#=
function HC_differentiate(f::SymEngine.Basic, v)
    a = SymEngine.Basic()
    ret = ccall(
        (:basic_diff, SymEngine.libsymengine),
        Int,
        (Ref{SymEngine.Basic}, Ref{SymEngine.Basic}, Ref{SymEngine.Basic}),
        a,
        f,
        v,
    )
    return a
end

function HC_jac(guess, P)
    XX = [SymEngine.symbols("xx_$i") for i in 1:P.num_vars]
    jac_sym = [HC_differentiate(symbolic_evaluate_waring_poly(XX, P), v) for v in XX]
    sub_guess = [XX[idx]=>guess[idx] for idx in 1:num_vars]
    for sub in sub_guess
        jac_sym .= [SymEngine.subs(e, sub) for e in jac_sym]
    end
    return jac_sym
end
println("HC:")
HC_res = @btimed jac_HC = HC_jac(guess, P)
println("value: $(HC_res.value)")
println("time: $(HC_res.time)")
println("alloc: $(HC_res.alloc)")
println()
=#
