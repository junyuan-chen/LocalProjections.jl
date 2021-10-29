module LocalProjections

using BSplines: BSplineBasis, basismatrix
using LinearAlgebra: cholesky!, inv!, svd!, ldiv!, mul!, I
using StatsBase: CovarianceEstimator, StatisticalModel, RegressionModel, CoefTable, TestStat
using StatsFuns: normccdf, norminvcdf
using Tables

import Base: ==, show, size, length, vec
import StatsBase: coef, vcov, stderror, confint, coeftable, modelmatrix, residuals
import Tables: getcolumn

# Reexport objects from StatsBase
export coef, vcov, stderror, confint, coeftable, modelmatrix, residuals

export hamilton_filter,
       TransformedVar,
       Cum,

       AbstractEstimator,
       LeastSquareLP,
       AbstractEstimatorResult,
       LocalProjectionResult,
       lp,

       SearchCriterion,
       LOOCV,
       GCV,
       AIC,
       SmoothAlgorithm,
       DemmlerReinsch,
       DirectSolve,
       ModelSelection,
       GridSearch,
       grid,
       ModelSelectionResult,
       GridSearchResult,
       SmoothLP,
       SmoothLPResult,

       SimpleVCE,
       HRVCE,

       ImpulseResponse,
       irf

include("utils.jl")
include("lp.jl")
include("slp.jl")
include("vce.jl")
include("irf.jl")

end # module
