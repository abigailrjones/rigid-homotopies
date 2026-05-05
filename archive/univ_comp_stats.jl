using DelimitedFiles
using Statistics

function rank_statistics(rank, data)
    num_rows = size(data)[1]
    #=
    for idx in 1:num_rows
        if data[idx,1] == rank
            max_idx = num_rows < idx+100 ? num_rows : idx+100
            arr = view(data, idx:max_idx, 3)
            return [max_idx+1-idx, mean(arr), var(arr), median(arr),
                    minimum(arr), maximum(arr)]
        end
    end
    println("Rank not found in inputted data.")
    =#
    return [num_rows, mean(data), var(data), median(data), minimum(data), maximum(data)]
end

function print_statistics(rank, num_samples, MEAN, VAR, MEDIAN, MIN, MAX)
    println("There are $num_samples entries with rank $rank.")
    println("mean (# of iterations): $MEAN")
    # println("variance (# of iterations): $VAR")
    # println("median (# of iterations): $MEDIAN")
    println("min (# of iterations): $MIN")
    println("max (# of iterations): $MAX")
    println("ratio (mean # of iterations / rank): $(MEAN/rank)")
    println("")
end

PREV_MEAN = 1.0
for rank in 4:10
    data = readdlm("examples/data/complexity_april/data_univ_complexity_$rank.txt")
    num_samples, MEAN, VAR, MEDIAN, MIN, MAX = rank_statistics(rank, data)
    println("Ratio of consecutive mean iterations: $(MEAN/PREV_MEAN)")
    global PREV_MEAN = MEAN
    print_statistics(rank, num_samples, MEAN, VAR, MEDIAN, MIN, MAX)
end


_, PREV_MEAN, _, _, _, _ = rank_statistics(10, readdlm("examples/data/complexity/data_univ_complexity_10.txt"))
for i in 2:6
    rank = 5*2^i
    data = readdlm("examples/data/complexity_april/data_univ_complexity_$rank.txt")
    num_samples, MEAN, VAR, MEDIAN, MIN, MAX = rank_statistics(rank, data)
    println("Ratio of consecutive mean iterations: $(MEAN/PREV_MEAN)")
    global PREV_MEAN = MEAN
    print_statistics(rank, num_samples, MEAN, VAR, MEDIAN, MIN, MAX)
end
