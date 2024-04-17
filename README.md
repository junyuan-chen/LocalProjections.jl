# LocalProjections.jl

*Local projection methods for impulse response estimation*

[![CI-stable][CI-stable-img]][CI-stable-url]
[![codecov][codecov-img]][codecov-url]
[![version][version-img]][version-url]
[![PkgEval][pkgeval-img]][pkgeval-url]

[CI-stable-img]: https://github.com/junyuan-chen/LocalProjections.jl/actions/workflows/CI-stable.yml/badge.svg?branch=main
[CI-stable-url]: https://github.com/junyuan-chen/LocalProjections.jl/actions/workflows/CI-stable.yml

[codecov-img]: https://codecov.io/gh/junyuan-chen/LocalProjections.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/junyuan-chen/LocalProjections.jl

[version-img]: https://juliahub.com/docs/LocalProjections/version.svg
[version-url]: https://juliahub.com/ui/Packages/LocalProjections/WJbkE

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/L/LocalProjections.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/L/LocalProjections.html

[LocalProjections.jl](https://github.com/junyuan-chen/LocalProjections.jl)
is a Julia package for estimating impulse response functions with local projection methods.
It follows the latest development in the econometric literature
and pursues reliable and efficient implementation.

## Features

A growing list of features includes the following:

- Ordinary least squares local projections as in [Jordà (2005)](https://doi.org/10.1257/0002828053828518)
- Instrumental variable (LP-IV) estimation and unit effect normalization as in [Stock and Watson (2018)](https://doi.org/10.1111/ecoj.12593)
- Cumulative effects and state dependency as in [Ramey and Zubairy (2018)](https://doi.org/10.1086/696277)
- Panel local projections with fixed effects as in [Jordà, Schularick and Taylor (2015)](https://dx.doi.org/10.1016/j.jinteco.2014.12.011)
- Smooth local projections as in [Barnichon and Brownlees (2019)](https://doi.org/10.1162/rest_a_00778)
- Equal-weighted cosine (EWC) test for HAR inference as in [Lazarus et al. (2018)](https://doi.org/10.1080/07350015.2018.1506926)

## Quick Start

Estimation for a given specification is conducted by calling the function `lp`.
An impulse response function with respect to a specific variable is then obtained via `irf`.
[Example data](data) are included in this package for convenience.

To reproduce the empirical illustration from
[Barnichon and Brownlees (2019)](https://doi.org/10.1162/rest_a_00778):

```julia
using LocalProjections, CSV
# Read example data from Barnichon and Brownlees (2019)
df = CSV.File(datafile(:bb))
# Specify names of the regressors
ns = (:ir, :pi, :yg)
# By default, conduct least-squares estimation
# Lags for the outcome variable and variables in `wnames` are included automatically
r_ls = lp(df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1)
# Collect the estimated impulse response function with respect to ir
f_ls = irf(r_ls, :yg, :ir)

# For smooth local projections, specify the estimator
est = SmoothLP(:ir, 3, 2, search=grid(194.0.*(1:0.5:10)), criterion=LOOCV())
r_sm = lp(est, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1)
f_sm = irf(r_sm, :yg, :ir)
```

To reproduce the cumulative multipliers from
[Ramey and Zubairy (2018)](https://doi.org/10.1086/696277):

```julia
using LocalProjections, CSV
# Read example data from Ramey and Zubairy (2018)
df = CSV.File(datafile(:rz))
# Replicate results in their Table 1
# Baseline linear specification with news shock
# `Cum()` is for cumulative effects
r1 = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>:newsy,
    nlag=4, nhorz=17, addylag=false, firststagebyhorz=true)
f1 = irf(r1, Cum(:y), Cum(:g))

# State-dependent specification with both news shock and Blanchard-Perotti shock
# Interactions with states are handled automatically for variables in `wnames`
# For other variables, interactions with states need to be constructed before calling `lp`
r2 = lp(df, Cum(:y), xnames=(Cum(:g,:rec), Cum(:g,:exp), :rec), wnames=(:newsy, :y, :g),
        iv=(Cum(:g,:rec), Cum(:g,:exp))=>(:recnewsy, :expnewsy, :recg, :expg),
        states=(:rec, :exp), nlag=4, nhorz=16, minhorz=1, addylag=false, firststagebyhorz=true)
# Collect the estimated impulse response functions in each state separately
f2rec = irf(r2, Cum(:y), Cum(:g,:rec))
f2exp = irf(r2, Cum(:y), Cum(:g,:exp))
```

For panel local projections, specify the variable identifying each unit:

```julia
using LocalProjections, CSV
# Example from Jordà's website using the macrohistory database
df = CSV.File(datafile(:jst))
ws = (:dlgrgdp, :dlgcpi, :dstir)
# Country fixed effects are included by default
# Compute cluster-robust standard errors via Vcov.jl
r = lp(df, :dlgrgdp, wnames=ws, nlag=3, nhorz=5, panelid=:iso, vce=cluster(:iso))
f = irf(r, :dlgrgdp, :dlgrgdp, lag=1)
```

For more details,
enter the [help](https://docs.julialang.org/en/v1/stdlib/REPL/#Help-mode) mode.

## References

**Barnichon, Regis, and Christian Brownlees.** 2019.
"Impulse Response Estimation by Smooth Local Projections."
*The Review of Economics and Statistics* 101 (3): 522-530.

**Jordà, Òscar.** 2005. "Estimation and Inference of Impulse Responses by Local Projections."
*American Economic Review* 95 (1): 161-182.

**Jordà, Òscar., Moritz Schularick, and Alan M. Taylor.** 2015.
"Betting the House."
*Journal of International Economics* 96 (S1): S2-S18.

**Lazarus, Eben, Daniel J. Lewis, James H. Stock, and Mark W. Watson.** 2018.
"HAR Inference: Recommendations for Practice."
*Journal of Business & Economic Statistics* 36 (4): 541-559.

**Li, Dake, Mikkel Plagborg-Møller, and Christian K. Wolf.** 2021.
"Local Projections vs. VARs: Lessons from Thousands of DGPs." Unpublished.

**Montiel Olea, José Luis, and Mikkel Plagborg-Møller.** 2021.
"Local Projection Inference is Simpler and More Robust Than You Think."
*Econometrica* 89 (4): 1789-1823.

**Plagborg-Møller, Mikkel, and Christian K. Wolf.** 2021.
"Local Projections and VARs Estimate the Same Impulse Responses."
*Econometrica* 89 (2): 955-980.

**Ramey, Valerie A., and Sarah Zubairy.** 2018.
"Government Spending Multipliers in Good Times and in Bad: Evidence from US Historical Data."
*Journal of Political Economy* 126 (2): 850-901.

**Stock, James H., and Mark W. Watson.** 2018.
"Identification and Estimation of Dynamic Causal Effects in Macroeconomics Using External Instruments."
*The Economic Journal* 128 (610): 917-948.
