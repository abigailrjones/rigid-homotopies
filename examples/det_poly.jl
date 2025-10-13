#=

Builds a system of determinantal polynomials.

=#

function rand_mat(dim, complex=false)
    if complex
        return rand(Float64,(dim,dim)) + im*rand(Float64,(dim,dim))
    else
        return rand(Float64,(dim,dim))
    end
end

function build_det_poly_system(degrees, num_vars)
    F = []
    for deg in degrees
        M = rand_mat(deg)
        push!(F, X -> det(sum([X[i]*M for i in 1:num_vars])))
    end
    return F
end

function random_plotting()
    p = plot()
    X = Y = -10:0.1:10
    for idx in 1:num_funcs
        f = (x,y) -> F[idx]([x,y])
        plot!(X, Y, f, st=:surface)
    end
    display(p)
end

