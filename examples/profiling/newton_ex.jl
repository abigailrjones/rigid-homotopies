include("../../utils.jl")

F = [X -> X[1]^2 + X[2]*X[3] - 2]#, X -> X[1]*X[2] + X[3]^3, X -> X[3]^4 + X[2]^4 + X[1]^2]
F_t = X -> [F[idx](X) for idx in 1:length(F)]
# @btime newton!(rand(ComplexF64, 3), F_t)
@profview for i in 1:10 newton!(rand(ComplexF64, 3), F_t) end
