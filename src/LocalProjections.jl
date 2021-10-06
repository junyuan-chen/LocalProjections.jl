module LocalProjections

using StatsBase: CovarianceEstimator, StatisticalModel, RegressionModel, CoefTable
using StatsFuns: normccdf, norminvcdf
using Tables
using Tables: getcolumn

import Base: show
import StatsBase: coef, vcov, stderror, confint, coeftable, modelmatrix, residuals

# Reexport objects from StatsBase
export coef, vcov, stderror, confint, coeftable, modelmatrix, residuals

export hamilton_filter,

       LocalProjectionResult,
       lp,

       SimpleVCE,
       HRVCE,

       ImpulseResponse,
       irf

include("utils.jl")
include("lp.jl")
include("vce.jl")
include("irf.jl")

end # module
