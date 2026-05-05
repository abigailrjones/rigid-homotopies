using DelimitedFiles
using Statistics

function rank_statistics(rank, data)
    num_rows = size(data)[1]
    for idx in 1:num_rows
        if data[idx,1] == rank
            max_idx = num_rows < idx+100 ? num_rows : idx+100
            arr = view(data, idx:max_idx, 3)
            return [max_idx+1-idx, mean(arr), var(arr), median(arr),
                    minimum(arr), maximum(arr)]
        end
    end
    println("Rank not found in inputted data.")
end

function print_statistics(rank, num_samples, MEAN, VAR, MEDIAN, MIN, MAX)
    println("There are $num_samples entries with rank $rank.")
    println("mean (# of iterations): $MEAN")
    # println("variance (# of iterations): $VAR")
    println("median (# of iterations): $MEDIAN")
    println("min (# of iterations): $MIN")
    println("max (# of iterations): $MAX")
    println("")
end

base = 2

for i in 2:4
    rank = base^i
    data = readdlm("examples/data/data_complexity.txt")
    num_samples, MEAN, VAR, MEDIAN, MIN, MAX = rank_statistics(rank, data)
    print_statistics(rank, num_samples, MEAN, VAR, MEDIAN, MIN, MAX)
end

for i in 5:8
    rank = base^i
    data = readdlm("examples/data/complexity/data_complexity_$rank.txt")
    num_samples, MEAN, VAR, MEDIAN, MIN, MAX = rank_statistics(rank, data)
    print_statistics(rank, num_samples, MEAN, VAR, MEDIAN, MIN, MAX)
end
