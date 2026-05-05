using BenchmarkTools: @btimed
import Zygote
using LinearAlgebra: I
using Enzyme
import SymEngine


include("../example_utils.jl")

#=
num_vars = 2
deg = 2
rank = 2
# 2 x 2 matrix (num_vars x rank)
M1 = [1. + 0*im 2.; 3. 4.]
# P(x,y) = 5x^2 + 22xy + 25y^2
P1 = WaringPoly(num_vars,deg,rank,M1)

M2 = [2. + 0*im 3.; 1. 3.]
P2 = WaringPoly(num_vars,deg,rank,M2)

guess = [1. + 1*im, 2.]

# true jacobian
G(x,y) = [10*x + 22*y 22*x + 50*y; 26*x + 22*y 22*x + 20*y]
println("True solution: $(G(guess[1],guess[2]))\n")
=#

num_vars = 5
num_funcs = num_vars - 1
deg = 5
rank = 5

guess = ones(ComplexF64, num_vars)

poly_system = Vector{WaringPoly}(undef, num_funcs)
for idx in 1:num_funcs
    poly_system[idx] = WaringPoly(num_vars, deg, rank)
end


# ZYGOTE
println("Zygote:")
F_t = X -> old_evaluate_waring_system(X, poly_system)
zyg_res = @btimed jac_zygote = Zygote.jacobian(x -> real(F_t(x)), guess)[1] |> conj
println("value: $(zyg_res.value[1])")
println("time: $(zyg_res.time)")
println("alloc: $(zyg_res.alloc)")
println()

# ENZYME
global id = (1.0+0*im)*Matrix(I, num_vars, num_vars)
global d_guess = Tuple(id[:,idx] for idx in 1:num_vars)
function build_jacobian(guess, params)
    num_vars = params[1].num_vars
    return Enzyme.autodiff(Forward, evaluate_waring_system, Duplicated,
                           BatchDuplicated(guess, d_guess), Const(params))
end

println("Enzyme:")
enzyme_res = @btimed jac_enzyme = build_jacobian(guess, poly_system)
println("value: $(enzyme_res.value[1][1][1])")
println("time: $(enzyme_res.time)")
println("alloc: $(enzyme_res.alloc)")
println()

#=
# SYMENGINE
function symbolic_jac(params)
    global XX = [SymEngine.symbols("xx_$i") for i in 1:params[1].num_vars]
    jac_sym = [SymEngine.diff(symbolic_evaluate_waring_poly(XX, P), v) for P in params for v in XX]
    return jac_sym
end

function symbolic_evaluation(guess, jac_sym)
    sub_guess = [XX[idx]=>guess[idx] for idx in 1:num_vars]
    for sub in sub_guess
        jac_sym .= [SymEngine.subs(e, sub) for e in jac_sym]
    end
    return jac_sym
end
println("Symbolic:")
sym_res = @btimed jac_sym = symbolic_jac(poly_system)
# println("value: $(sym_res.value)")
println("time: $(sym_res.time)")
println("alloc: $(sym_res.alloc)")

sym_res = @btimed jac_sym = symbolic_evaluation(guess, sym_res.value)
println("value: $(sym_res.value[1])")
println("time: $(sym_res.time)")
println("alloc: $(sym_res.alloc)")
println()

# MATRIX MULTIPLICATION
sym_jac = reshape(sym_res.value, (num_funcs, num_vars))
mat_res = @btimed jac_mat_mult = sym_jac * rand(ComplexF64, num_vars, num_funcs)
println("value: $(mat_res.value[1])")
println("time: $(mat_res.time)")
println("alloc: $(mat_res.alloc)")
=#
