To build Table 2, we must (1) compute the data, and (2) print the number
of successes.

(1) To compute the data, make sure to set `num_vars', `degrees', and
`rank' in `heuristic.jl' to the chosen values. Then run `heuristic.jl'
in this directory.

Note: currently `mid_print' is set to true so that we know how far the
computation has gotten. If you do not wish to see this, you can set it
to false.

Note: `val' determines the heuristic step size, as dt = 10^(-val). You
may change the range of val; this value is also used to differentiate
filenames.

BE CAREFUL ABOUT OVERWRITING EXISTING FILES. To this end, there is an
`@assert false' flag at the top of `heuristic.jl', which writes to
files.

(2) To print some statistics and see the number of successes, run
`print_heuristics.jl' from within this directory. Make sure to set
`rank' to be the same in this file as it was for `heuristic.jl'.
