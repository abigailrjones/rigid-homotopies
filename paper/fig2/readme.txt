To build figure 2, we need to (a) build the table, and (b) build the
associated bar plot.

(a) To build the table, we need to (1) compute the data, (2) compute the
averages and medians of the computed data, and (3) build the LaTeX code
for the resulting table of values.

(1) Hopefully this is already done, but if you have to rerun and get new
data, this can be done more efficiently by using GNU Parallel. In
particular, if you run

    parallel julia paper/fig2/track_data.jl ::: 2 ::: 3 ::: 4 5

from within the `rigid-homotopies' directory, then `track_data.jl' is
run with arguments (2,3,4) and (2,3,5). Choose a set of arguments that
span the parameter space you care about and run the associated GNU
parallel command. Note: to make sure you are actually getting the set
of arguments you wanted to get, you can always run

    parallel echo ::: 2 ::: 3 ::: 4 5

first. In this example, the output should be

    2 3 4
    2 3 5.

For the current table, the arguments are

    parallel julia paper/fig2/track_data.jl ::: 2 3 ::: 3 4 5 ::: 3 4 5.

If you wish to reuse the data already computed, it can be found in
`data'.

Note that, if you want to check the number of instances that have
completed, you may run something like

    parallel julia count_data.jl ::: 2 3 ::: 3 4 5 ::: 3 4 5

from within this directory to see how many instances have completed.

BE CAREFUL ABOUT OVERWRITING EXISTING FILES. To this end, there is an
`@assert false' flag at the top of `track_data.jl', which writes to
files.

(2-3) To compute the average and median values needed for the table, we
run `track_data_statistics.jl'. However, we also want to write these
values to a file so that they are LaTeX-ready. To this end, we run

    bash write_table.sh

on the command line (from within the current directory). Note that this
file is hard-coded for the arguments given above. If you wish to
change them, simply change the long list of arguments following the
while loop in `write_table.sh'. Make sure your arguments are in the
order you wish them to be in your table.

Running `write_table.sh' writes the LaTeX-ready values to a file in
this directory called `data_table1.txt'. You can directly copy and paste
the contents of this file into a tabular environment in LaTeX with nine
columns to build the table.

(b) To build the bar plot, we need to (1) compute the data, (2) compute
the averages and medians of the computed data, and (3) build the bar
plot.

(1) We reuse the data computed to build the table.

(2) We reuse the script `write_table.sh`, which calls
`track_data_statistics.jl`. However, we have to change the boolean
`uselatex` in `track_data_statistics.jl` from `true` to `false` so that
we store the data directly without any LaTeX formatting.

(3) Then we run `julia plot_table_bar.jl`, and the bar plot will be
stored in the file `table_bar.png`.
