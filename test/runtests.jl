using Test
using LocalProjections

using CSV
using DataFrames
using FixedEffectModels
using LinearAlgebra: I
using LocalProjections: kron_fastl, kron_fastr, getscore, _geto,
    OLS, VarName, _makeYX, _firststage, _lp, _toname,
    Ridge, _basismatrix, _makeYSr, _makeP
using ShiftedArrays: lag
using Tables: getcolumn

exampledata(name::Union{Symbol,String}) = CSV.read(datafile(name), DataFrame)

const tests = [
    "utils",
    "data",
    "lp",
    "slp",
    "vce",
    "irf"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
