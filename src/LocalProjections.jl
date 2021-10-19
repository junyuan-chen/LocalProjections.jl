module LocalProjections

using StatsBase: CovarianceEstimator, StatisticalModel, RegressionModel, CoefTable
using StatsFuns: normccdf, norminvcdf
using Tables

import Base: show, size, length, vec
import StatsBase: coef, vcov, stderror, confint, coeftable, modelmatrix, residuals
import Tables: getcolumn

# Reexport objects from StatsBase
export coef, vcov, stderror, confint, coeftable, modelmatrix, residuals

export hamilton_filter,
       TransformedVar,
       Cum,

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
