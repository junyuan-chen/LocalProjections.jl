module LocalProjections

using BSplines: BSplineBasis, basismatrix
using FFTW: fft!
using FixedEffectModels: AbstractFixedEffectSolver, Combination, FixedEffect,
    solve_residuals!, isnested, nunique
using GroupedArrays: GroupedArray
using LinearAlgebra: I, Symmetric, cholesky!, svd!, ldiv!, inv!, mul!
using StatsAPI: RegressionModel, StatisticalModel
using StatsBase: CovarianceEstimator, CoefTable, TestStat,
    AbstractWeights, Weights, UnitWeights, uweights
using StatsFuns: tdistccdf, tdistinvccdf, chisqccdf
using Tables
using Vcov: ClusterCovariance, VcovData, robust, cluster, names, nclusters,
    ranktest!, pinvertible

import Base: ==, show, size, length, vec, view
import StatsAPI: coef, vcov, stderror, confint, coeftable, modelmatrix, residuals, dof_residual
import Tables: getcolumn
import Vcov: S_hat, dof_tstat

# Reexport objects from StatsAPI
export coef, vcov, stderror, confint, coeftable, modelmatrix, residuals, dof_residual
# Reexport objects from Vcov
export cluster, dof_tstat

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
