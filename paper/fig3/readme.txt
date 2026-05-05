To build Figure 3, run `plot_zero_paths.jl' from within the directory
with the associated .txt files containing the data. You can change the
data that is used by changing the filenames that are passed in.

This will output plots called `filename.png' in the current directory.

There will be too much whitespace, mostly on the left side. To get rid
of this, run

    convert -trim filename.png filename.png

on the command line.

To rerun or get data for different configurations, build a Waring system
with the desired configuration and then pass to `solve' with the
optional argument `filename=paper/tab1/filename.txt' filled in. (Note
that this path is only correct if `solve' is being called from the
`rigid-homotopies' directory.) This will store the information needed
for intermediate plotting in a file of the given name, which can then be
passed to `plot_zero_paths.jl'.
