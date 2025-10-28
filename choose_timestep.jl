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

using Zygote: gradient, jacobian
using FFTW: fft

# Note to self: verified real version via plotting
function sample_unit_ball(dim::Integer, num_pts::Integer)
    x = randn(ComplexF64,(num_pts,dim))
    r = rand(Float64,(num_pts,)).^(1/dim)
    norm = sqrt.(sum(abs2, x; dims=2))
    return [x[i,:]*r[i]/norm[i] for i in 1:num_pts]
end

function estimate_gammaprob(f::Function,Z,eta,D::Integer,num_vars)
    h(X) = f(Z + X)
    # compute squared norm of gradient of h at zero vector. Note: for
    # holomorphic functions (like polynomials), the Cauchy Riemann equations
    # are satisfied, so complex derviative can be written as the complex
    # conjugate of the gradient of the real part (see Zygote ``Complex
    # Differentiation" documentation for more details)
    d0h_sq_norm = sum(abs.(gradient(X->real(h(X)), zeros(num_vars))[1] |> conj).^2)
    s = ceil(Int64, 1 + log(2, D/eta))
    rand_w = sample_unit_ball(num_vars, s)
    sum_squared_components = zeros(D+1)
    for w in rand_w
        # compute each degree component of h evaluated at w, square elementwise
        # and add to sum
        sum_squared_components += abs.(compute_deg_components(h,w,D)).^2
    end
    return_est = 0
    fac = k -> (32*(num_vars-1)*k)^k*binomial(num_vars+k, k)/d0h_sq_norm/s
    for k in 2:D
        # FIXME sometimes fac(k) gets big and negative and this produces a
        # DomainError with the exponentiation (something to do with complex)
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
    return fft([f(exp(2*pi*im*j/(D+1))*input) for j in 0:D]) / (D+1)
end

# condition number is denoted by \kappa in the papers
# in particular, we consider the L2 norm of the matrix L(F_t, z) (defined in
# (15) in RH1), which is the inverse of the smallest singular value of L
function compute_condition_num(F, W_t, Z, num_funcs)
    F_t = X -> [F[idx](W_t[idx]' * X) for idx in 1:num_funcs]
    jac = jacobian(x -> real(F_t(x)), Z)[1] |> conj
    row_norms = 1.0 ./ sqrt.(sum(abs2, jac; dims=2))
    L = jac .* row_norms
    # note that svdvals orders singular values in descending order
    return 1.0 / svdvals(L)[end]
end

function choose_timestep(F, W_t, Z, D, max_iter, num_funcs, num_vars; eps=1e-8)
    sum_g_sq = 0.0
    for idx in 1:num_funcs
        func = F[idx]
        W = W_t[idx]
        sum_g_sq += estimate_gammaprob(X -> func(W' * X),Z,eps/((num_vars-1)*max_iter),D,num_vars)^2
    end
    return 1/(240 * compute_condition_num(F, W_t, Z, num_funcs)^2 * sqrt(sum_g_sq))
end
