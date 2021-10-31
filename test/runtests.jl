using Test
using LocalProjections

using CSV
using CodecZlib: GzipDecompressorStream
using DataFrames
using LinearAlgebra: I
using LocalProjections: kron_fastl, kron_fastr, getscore, _geto,
    OLS, VarName, _makeYX, _firststage, _lp, _toint, _toname,
    Ridge, _basismatrix, _makeYSr, _makeP
using Tables: getcolumn

function exampledata(name::Union{Symbol,String})
    # Need to allow CSV v0.8 to work
    f = open(datafile(name)) |> GzipDecompressorStream |> read |> CSV.File
    return DataFrame(f, copycols=true)
end

const tests = [
    "utils",
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
