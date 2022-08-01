module LocalProjections

using BSplines: BSplineBasis, basismatrix
using FFTW: fft!
using FixedEffectModels: AbstractFixedEffectSolver, Combination, FixedEffect, solve_residuals!
using LinearAlgebra: I, cholesky!, svd!, ldiv!, inv!, mul!
using StatsBase: CovarianceEstimator, RegressionModel, StatisticalModel, CoefTable, TestStat,
    AbstractWeights, Weights, UnitWeights, uweights
using StatsFuns: normccdf, norminvccdf, tdistccdf, tdistinvccdf
using Tables

import Base: ==, show, size, length, vec, view
import StatsBase: coef, vcov, stderror, confint, coeftable, modelmatrix, residuals
import Tables: getcolumn

# Reexport objects from StatsBase
export coef, vcov, stderror, confint, coeftable, modelmatrix, residuals

export hamilton_filter,
       TransformedVar,
       Cum,
       datafile,

       LPData,

       OLS,
       AbstractEstimator,
       LeastSquaresLP,
       AbstractEstimatorResult,
       LeastSquaresLPResult,
       LocalProjectionResult,
       lp,

       Ridge,
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
       LongRunVariance,
       HARVCE,
       EqualWeightedCosine,
       EWC,
       criticalvalue,
       pvalue,

       ImpulseResponse,
       irf

include("utils.jl")
include("data.jl")
include("lp.jl")
include("slp.jl")
include("vce.jl")
include("irf.jl")

end # module
