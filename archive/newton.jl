include("../utils.jl")
using Plots

num_vars = 1
# degrees = [3, 3]
# ranks = degrees .+ 1
# P = build_waring_system(num_vars, degrees, ranks)


# P = [X -> X[1]^2 - 1, X -> X[1]^3 + X[1]^2]
P = [X -> X[1]^4 + X[1]^3 - X[1]^2 + 4]

root = rand(ComplexF64, num_vars)#*(-1)
prev_root = copy(root)
println("Initial root is $root.")

num_iter = newton!(root, P)
println("Converged to $root in $num_iter steps.")

MAXITER = 20
p = plot()

num_iter = 0
success = false
root = copy(prev_root)
while !success && num_iter < MAXITER
    try
        _ = newton!(root, P, max_iter=1)
    catch
        # println("$(root)")# .- prev_root)")
        plot!([num_iter],abs.(real.(root .- prev_root)),seriestype=:scatter)
        prev_root .= root
        global num_iter += 1
    else
        plot!([num_iter],abs.(real.(root .- prev_root)),seriestype=:scatter)
        global success = true
        global num_iter += 1
        println("Converged to $root in $num_iter steps.")
    end
end

if num_iter >= MAXITER println("Failed to converge within $MAXITER iterations.") end
x = range(1,num_iter,length=100)
y = (1 ./ (x .^2))
plot!(x,y)
savefig(p, "newton_plot.png")
# display(p)
