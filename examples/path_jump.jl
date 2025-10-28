include("../rigid_hom.jl")
include("example_utils.jl")
include("../utils.jl")

TOL = eps()*1000

open("examples/data/data_path_jump.txt", "w") do f
    # empty file of previous contents
end

num_vars = 4
degrees = [4,4,4]
num_funcs = length(degrees)
max_degree = maximum(degrees)
# max_iter defined below, based on initial_dt
use_heuristic = true
mid_print = false

for idx in 1:1000
    F = build_my_system(degrees, num_vars)

    # build start system and path here, so that randomness is fixed throughout the example
    start_system, start_root = build_start_system(F, num_vars)
    check_build_start_system(F, start_system, start_root, num_funcs)
    path = build_path(start_system)
    check_build_path(path, start_system, num_vars)

    dts = [0.1,0.01,0.001]#,0.0001]#,0.00001]
    roots = Dict()
    for initial_dt in dts
        final_root, _ = solve(F, num_funcs, num_vars, max_degree,
                              ceil(Int,1/initial_dt)+1, start_system,
                              start_root, path; use_heuristic=use_heuristic,
                              mid_print=mid_print, initial_dt=initial_dt)

        compare_zero!(roots, final_root, num_vars)

        #=
        FF = X -> [F[idx](X) for idx in 1:num_funcs]
        refined_root, refine_count = refine(final_root, FF)
        refined_root = refined_root./refined_root[1]
        println()
        println("After $refine_count iterations, final root was refined by $(abs.(final_root - refined_root))")
        println("Final root: $refined_root")
        println("System residuals: $(FF(refined_root))")
        =#

        #=
        open("examples/data/data_path_jump.txt", "a") do f
            write(f, "$initial_dt $final_root\n")
        end
        =#
    end

    open("examples/data/data_path_jump.txt", "a") do f
        write(f, "$(length(roots)==1)\n")
    end
end
