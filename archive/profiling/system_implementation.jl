using Enzyme
using BenchmarkTools

function build_gradient_reverse!(guess, P, val)
    res = zeros(ComplexF64, P.num_vars)
    val .= Enzyme.autodiff(ReverseHolomorphicWithPrimal, P.func, Active, Duplicated(guess, res))[end]
    return res
end

function build_jacobian_reverse(guess, params)
    # TODO can we just have one allocation for jac the whole time and pass it in?
    jac = zeros(ComplexF64, length(params), params[1].num_vars)
    vals = zeros(ComplexF64, length(params))
    for idx in 1:length(params)
        P = params[idx]
        view(jac,idx,:) .= build_gradient_reverse!(guess, P, view(vals,idx))
    end
    return jac, vals
end

function build_gradient_reverse!(guess, func, num_vars, val)
    res = zeros(ComplexF64, num_vars)
    val .= Enzyme.autodiff(ReverseHolomorphicWithPrimal, func, Active, Duplicated(guess, res))[end]
    return res
end

function build_jacobian_reverse(guess, funcs, num_vars)
    # TODO can we just have one allocation for jac the whole time and pass it in?
    jac = zeros(ComplexF64, length(funcs), num_vars)
    vals = zeros(ComplexF64, length(funcs))
    for idx in 1:length(funcs)
        view(jac,idx,:) .= build_gradient_reverse!(guess, funcs[idx], num_vars, view(vals,idx))
    end
    return jac, vals
end

NUM_VARS::Int = 10
guess = rand(ComplexF64, NUM_VARS)

# case 1: vector of anonymous functions
F1 = [x -> sum(x .^ 2), x -> sum(x .^ 3)]
# res1 = @btimed build_jacobian_reverse(guess, F1, NUM_VARS)
# println("time=$(res1.time), alloc=$(res1.alloc)")

# case 2: vector of non-parametric Poly structs
struct Poly
    num_vars::Int
    degree::Int
    func::Function
end

F2 = [Poly(NUM_VARS, 2, x -> sum(x .^ 2)), Poly(NUM_VARS, 3, x -> sum(x .^ 3))]
# res2 = @btimed build_jacobian_reverse(guess, F2)
# println("time=$(res2.time), alloc=$(res2.alloc)")

# case 3: vector of parametric Poly structs
struct PPoly{F}
    num_vars::Int
    degree::Int
    func::F
end

F3 = [PPoly(NUM_VARS, 2, x -> sum(x .^ 2)), PPoly(NUM_VARS, 3, x -> sum(x .^ 3))]
# res3 = @btimed build_jacobian_reverse(guess, F3)
# println("time=$(res3.time), alloc=$(res3.alloc)")
