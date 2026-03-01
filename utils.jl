using FFTW: fft!
using Enzyme
using LinearAlgebra: pinv, norm

CUT_OFF = 5.0

##############################################

struct WaringPoly
    num_vars::Int
    deg::Int
    rank::Int
    # can we include that M is a (rank x num_vars) dimensioned array?
    M::Array{ComplexF64}
end

# WaringPoly(num_vars,deg,rank) = WaringPoly(num_vars,deg,rank,rand(ComplexF64,num_vars,rank))
WaringPoly(num_vars,deg,rank) = WaringPoly(num_vars,deg,rank,rand(ComplexF64,rank,num_vars))

function (poly::WaringPoly)(X)::ComplexF64
    # return sum((poly.M * X).^poly.deg)
    res = 0.0 + 0*im
    for idx in 1:poly.rank
        res += sum(poly.M[idx,:] .* X)^poly.deg
    end
    return res
end

function build_waring_system(num_vars, degrees, ranks)
    system = [WaringPoly(num_vars, degrees[1], ranks[1])]
    for idx in 2:length(degrees)
        push!(system, WaringPoly(num_vars, degrees[idx], ranks[idx]))
    end
    return system
end

##############################################

function check_inputs(num_funcs, num_vars)
    if (num_funcs != num_vars-1)
        throw(ErrorException("Number of functions must equal number of non-homogeneous variables."))
    end
end

# returns a vector with D+1 components, representing the 0:Dth degree
# components of the given polynomial f evaluated at the input
function compute_deg_components!(component_array,func,input::ComplexF64,D::Integer)
    for idx in 0:D
        component_array[idx+1] = func(input)
        input *= exp(2*pi*im/(D+1))
    end
    fft!(component_array)
    return
end

function compute_deg_components!(component_array,func,input::Vector{ComplexF64},D::Integer)
    # return fft([func(exp(2*pi*im*j/(D+1))*input) for j in 0:D]) / (D+1)
    for idx in 0:D
        # component_array[idx+1] = func(exp(2*pi*im*idx/(D+1))*input)
        component_array[idx+1] = func(input)
        input .*= exp(2*pi*im/(D+1))
    end
    fft!(component_array)
    return
end

# TODO add option for additional constant arguments to func
function build_gradient_reverse!(output, input, func)
    grad = zeros(ComplexF64, length(input))
    output .= Enzyme.autodiff(ReverseHolomorphicWithPrimal, Const(func), Active,
                              Duplicated(input, grad))[end]
    return grad
end

# TODO add option for additional constant arguments to func
function build_gradient_reverse!(output, input, func, mat)
    shifted_input = mat * input
    grad = zeros(ComplexF64, length(shifted_input))
    output .= Enzyme.autodiff(ReverseHolomorphicWithPrimal, Const(func), Active,
                              Duplicated(shifted_input, grad))[end]
    # transpose(conjugate transpose) = conjugate
    return transpose(mat) * grad
end

function build_jacobian_reverse(input, system, matrices=nothing)
    # note that length(input) == num_vars
    jac = zeros(ComplexF64, length(system), length(input))
    output = zeros(ComplexF64, length(system))
    if matrices == nothing
        for idx in 1:length(system)
            view(jac,idx,:) .= build_gradient_reverse!(view(output,idx), input,
                                                       system[idx])
        end
    else
        for idx in 1:length(system)
            view(jac,idx,:) .= build_gradient_reverse!(view(output,idx), input,
                                                       system[idx], matrices[idx])
        end
    end
    return jac, output
end

function newton!(guess::Vector{ComplexF64}, system, matrices=nothing; max_iter=1000, tol=eps()*100)::Int
    inc = Vector{ComplexF64}(undef, length(guess))
    residual = 1.0
    err = 1.0
    give_up = 1.0e10
    num_iter = 0
    num_small = 0
    magic = 3

    while (err < give_up) & (num_iter < max_iter)
        if (num_small > magic) & isapprox(residual, 0.0, atol=tol)
            return num_iter
        else
            jac, output = build_jacobian_reverse(guess, system, matrices)
            inc .= (pinv(jac) * output)
            guess .-= inc
            residual = norm(output)
            err = norm(inc)

            if isapprox(err, 0.0, atol=tol) num_small += 1
            else num_small = 0
            end
            #=
            println("Iteration $num_iter: guess=$(round.(guess; digits=3)), next
                    guess=$(round.(next_guess; digits=3))")
            =#
            num_iter += 1
        end
    end

    if err >= give_up
        throw(ErrorException("Newton's method is going to infinity."))
        # return -1
    end

    if num_iter >= max_iter
        throw(ErrorException("Newton's method failed to converge within $max_iter iterations."))
        # return -2
    end
end

function scale_root!(x)
    max_elt = maximum(abs.(x))
    if max_elt >= CUT_OFF
        x ./= max_elt
    end
end

# TODO
function print_input()
end

function print_output(success, system, W_0, final_root, use_heuristic,
                      num_steps, avg_step_size, avg_newton_iters)
    println("")
    println("$(use_heuristic ? "Using a heuristic timestep..." : "Using a rigorous timestep...")")
    if success
        println("Converged in $num_steps step(s)")
    else
        println("Failed to converge in $num_steps step(s)")
    end
    println("Average timestep: $avg_step_size")
    println("Average number of Newton iterations per step: $avg_newton_iters")
    println("")
    if success
        println("Final root: $(round.(final_root; digits=8))")
        println("Final root (unrounded): $final_root")
        # println("System residuals: $(round.(target_system(final_root); digits=20))")
        println("System residuals: $([system[idx](W_0[idx] * final_root) for
                                      idx in 1:length(system)])")
    end
    return
end

function check_init_roots(system, init_roots, num_vars)
    for idx in 1:length(system)
        @assert isapprox(system[idx](init_roots[idx]), 0.0, atol=TOL)
        @assert (length(init_roots[idx]) == num_vars)
    end
end

function check_sampled_init_root(func, init_root)
    if (!isapprox(func(init_root), 0.0, atol=TOL))
        println("Not close enough to zero pre-scaling: $(func(init_root))")
    end
    @assert isapprox(func(init_root), 0.0, atol=TOL)
    if (!isapprox(func(init_root/norm(init_root)), 0.0, atol=TOL))
        println("Norm of proposed zero: $(norm(init_root))")
        println("Evaluation post-scaling (not close enough to zero): $(func(init_root/norm(init_root)))")
    end
    @assert isapprox(func(init_root/norm(init_root)), 0.0, atol=TOL)
end

# build_start_system will only work as expected if inputted polynomials are
# homogeneous (the process with unitary matrices fails otherwise); if this
# assert fails, it could mean that the inputted polynomials did not satisfy
# this requirement
function check_build_start_system(system, start_system, start_root, num_funcs)
    for idx in 1:num_funcs
        func = system[idx]
        U = start_system[idx]
        @assert isapprox(func(U' * start_root), 0.0, atol=TOL)
    end
    return
end

function check_build_path(path, start_system, num_vars)
    W_0 = path(0.0)
    W_1 = path(1.0)
    for idx in 1:length(W_0)
        @assert isapprox(W_0[idx]-I, zeros(num_vars,num_vars), atol=TOL)
        @assert isapprox(W_1[idx]'-start_system[idx], zeros(num_vars,num_vars), atol=TOL)
    end
    return
end

function check_track_path(system, W_0, final_root)
    for idx in 1:length(system)
        @assert isapprox(system[idx](W_0[idx] * final_root), 0.0, atol=TOL)
    end
    return
end
