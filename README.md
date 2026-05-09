# About the project

An implementation of rigid homotopy continuation[^1][^2][^3].

[^1]: Lairez (2020), [Rigid continuation paths I. Quasilinear average complexity for solving polynomial systems](https://doi.org/10.1090/jams/938)
[^2]: B&uuml;rgisser, Cucker, Lairez (2023), [Rigid continuation paths II. Structured polynomial systems](https://doi.org/10.1017/fmp.2023.7)
[^3]: Jones, Lee, Rodriguez (preprint), [Rigid homotopies for sampling from algebraic varieties: a Waring structure complexity model](https://arxiv.org/abs/2605.04302)


# Usage

## A Waring-type system

To run, start the REPL and include `rigid_hom.jl` (make sure you are within the directory containing this file).

To build a random Waring-type polynomial `f` with $n$ variables, degree $d$, and length $r$, run `f = WaringPoly(n, d, r)`. This will choose the coefficient matrix randomly.

To build a Waring-type polynomial `g` with a specific coefficient matrix $M$, run `g = WaringPoly(n, d, r, M)`, where $M$ is a matrix with dimensions $r \times n$ and element-type `ComplexF64`.

To build a random Waring-type system `F` without having to build each polynomial in the system by hand, run `F = build_waring_sytem(n, list-of-degrees, list-of-lengths)`. For example, `build_waring_system(3, [2, 4], [3, 5])` builds a random system containing two polynomials: the first has 3 variables, is degree 2, and has length 3; the second has 3 variables, is degree 4, and has length 5.

>[!NOTE]
>The implementation requires that there are $n-1$ polynomials in the system, if $n$ is the number of variables.

To compute a root of a system, we run

```
solve(system::Vector, number-of-functions::Int, number-of-variables::Int, list-of-degrees::Vector{Int}, max-iteration-count::Int).
```

`solve` returns an array containing: the final root, number of iterations, minimum step size, maximum step size, average step size, minimum $\hat{\gamma}\_{\text{Frob}}$, maximum $\hat{\gamma}\_{\text{Frob}}$, average $\hat{\gamma}\_{\text{Frob}}$, minimum condition number, maximum condition number, and average condition number.

<!-- This call samples a start root for the given system randomly, and constructs the default path. To pass in non-default values for these, TODO. -->

Optional arguments for `solve`.
* `filename::String` -- prints intermediate output to the given file
* `mid_print::Bool` -- if true (default is false), initialization information and success/failure information is printed to the screen
* `use_heuristic::Bool` -- if true (default is false), uses `initial_dt` as step size
* `initial_dt::Float64` -- the step size (default is 0.01), only used if `use_heuristic=true`


### Example

We first build a random system `F`.

```
num_vars = 3
num_funcs = 2
degrees = [2, 4]
lengths = [3, 5]
F = build_waring_sytem(num_vars, degrees, lengths)
```

We run `solve`, setting the maximum iteration count to 100.

```
solve(F, num_funcs, num_vars, degrees, 100)

Track path was unsuccessful.
ERROR: Rigid homotopy failed to converge in 100 iterations.
```

The computation gives an error. We run again, with `mid_print=true`.

```
solve(F, num_funcs, num_vars, degrees, 100, mid_print=true)

The default random start system and start root will be used, as well as the default path.
Track path was unsuccessful.
Using a rigorous timestep...
Failed to converge in 100 step(s)
Average timestep: 3.234079512328281e-6
Average number of Newton iterations per step: 5.0
ERROR: Rigid homotopy failed to converge in 100 iterations.
```

The computation fails again (as expected), but we see that the average step size over the first 100 iterations is $3.234 \times 10^{-6}$. This means that to travel along this particular path, we need to take at least 310,000 steps to converge.

However, since a new start root and a new path are chosen each time we run the program, we don't want to set our maximum iteration count based on a single example. Instead, we rerun the above several times and see that the average step sizes are all on the order of $10^{-6}$. Because of this, we set the maximum iteration count to 1,000,000.

```
solve(F, num_funcs, num_vars, degrees, 1000000, mid_print=true)

The default random start system and start root will be used, as well as the default path.

Using a rigorous timestep...
Converged in 236951 step(s)
Average timestep: 4.220288348604722e-6
Average number of Newton iterations per step: 5.0
Final root: ComplexF64[0.65530486 + 0.4251779im, 0.31100764 + 0.49342514im, -0.04550758 - 0.21791165im]
Final root (unrounded): ComplexF64[0.6553048590549918 + 0.42517789561898917im, 0.3110076367009555 + 0.4934251423168388im, -0.04550757500299956 - 0.21791164560470827im]
System residuals: ComplexF64[4.440892098500626e-16 + 1.1102230246251565e-16im, 2.220446049250313e-16 + 1.5959455978986625e-16im]
```

This time, the program terminates successfully (though it will likely take over 10 minutes to complete).

We can also rerun using a heuristic step size of 0.01.

```
solve(F, num_funcs, num_vars, degrees, 101, use_heuristic=true, initial_dt=0.01, mid_print=true)

The default random start system and start root will be used, as well as the default path.

Using a heuristic timestep...
Converged in 100 step(s)
Average timestep: 0.01
Average number of Newton iterations per step: 6.14

Final root: ComplexF64[-0.174852 - 0.78418995im, 0.31004042 - 0.29445959im, -0.17458532 + 0.0426638im]
Final root (unrounded): ComplexF64[-0.1748520032464971 - 0.7841899457272691im, 0.3100404174527053 - 0.29445958745382755im, -0.17458531901448898 + 0.04266379522178667im]
System residuals: ComplexF64[-1.1102230246251565e-16 + 1.1102230246251565e-16im, 4.996003610813204e-16 - 2.636779683484747e-16im]
```

The program terminates successfully, and only takes about a second to complete.


## An arbitrary system

You may also pass a custom polynomial system to `solve`. For the program to run at all, the system must

1. be homogeneous and
2. have $n$ variables and $n-1$ polynomials.

Passing arbitrary systems has not been thoroughly tested, so be aware that odd behavior is possible.

>[!NOTE]
>Enzyme can be very particular about the format of functions it is asked to differentiate. When providing an arbitrary system, you may run into failures triggered by the way your system is inputted. Be sure to read [the associated documentation](https://enzyme.mit.edu/index.fcgi/julia/stable/) for help with this. (Two common issues that we encountered involve the use of [temporary storage](https://enzyme.mit.edu/index.fcgi/julia/stable/faq/#Activity-of-temporary-storage) and [possible runtime activity](https://enzyme.mit.edu/index.fcgi/julia/stable/faq/#faq-runtime-activity).)

<!-- Example? TODO. -->


# A few installation details

This project was written and tested with Julia 1.10.10. Newer versions of Julia will be tested for compatibility soon.

Required Julia packages include: Enzyme, FFTW, LinearAlgebra, and Statistics.

Example files may require additional packages, but these are not necessary to run the primary program. These packages are: Colors, DelimitedFiles, LaTeXStrings, Plots, Plots.PlotMeasures, and Printf.
