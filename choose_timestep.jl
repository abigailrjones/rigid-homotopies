
#=
   JULIA STYLE GUIDE NOTES
   * apparently we should prefer to keep typing as non-specific as possible
   * if a function modifies its arguments, append ! to name of function
   * apparently they dislike underscores, and prefer to squish words together in
     function names whenever possible (bizarre, and I disagree with this one vehemently
     so I will be ignoring it)

   RANDOM NOTES
   * It probably makes sense to write this last step as a for loop, in
     fact we should probably minimize the number of list comprehensions we
     use, for clarity.
   * Also I wonder if it's possible to store the system as a struct of some kind,
     with num_funcs or num_vars as global elements or something?
=#

using Zygote: gradient

# Note to self: verified real version via plotting
function sample_unit_ball(dim::Integer, num_pts::Integer)
    x = randn(Float64,(num_pts,dim)) + im*randn(Float64,(num_pts,dim))
    r = rand(Float64,(num_pts,)).^(1/dim)
    norm = sqrt.(sum(abs.(x).^2, dims=2)) # dims=2 sums along rows
    return [x[i,:]*r[i]/norm[i] for i in 1:num_pts]
end

function estimate_gammaprob(f::Function,Z,eps,D::Integer)
    h(X::Vector) = f(Z + X...)
    # compute squared norm of gradient of h at zero vector. Note: for
    # holomorphic functions (like polynomials), the Cauchy Riemann equations
    # are satisfied, so complex derviative can be written as the complex
    # conjugate of the gradient of the real part (see Zygote ``Complex
    # Differentiation" documentation for more details)
    d0h_sq_norm = sum(abs.(gradient(X->real(h(X)), zeros(num_vars))[1] |> conj).^2)
    s = ceil(1 + log(2, D/eps))
    rand_w = sample_unit_ball(num_vars, s)
    sum_squared_components = zeros(D+1)
    for w in rand_w
        # compute each degree component of h evaluated at w, square elementwise
        # and add to sum
        sum_squared_components += abs.(compute_deg_components(h,w)).^2
    end
    return_est = 0
    fac = k -> (32*(num_vars-1)*k)^k*binomial(num_vars+k, k)/d0h_sq_norm/s
    for k in 2:D
        new_est = (fac(k)*sum_squared_components[k+1])^(1/(2*k-2))
        if new_est > return_est
            return_est = new_est
        end
    end
    return return_est
end

# returns a vector with D+1 components, representing the 0:Dth degree
# components of the given polynomial f evaluated at the input
function compute_deg_components(f::Function,input,D::Integer)
    return fft([f(exp(2*pi*im*j/(D+1))*input...) for j in 0:D]) / (D+1)
end

# TODO
function compute_kappa()
end

# TODO
function choose_timestep()
    for idx in 1:num_funcs
        w = Wt[idx]
        sum_g_sq += estimate_gammaprob(f circ w_inv,z,eps)^2
    end
    return 1/(240 * compute_kappa()^2 * sqrt(sum_g_sq))
end
