using Test
using LocalProjections

using CSV
using CodecZlib: GzipDecompressorStream
using DataFrames
using LinearAlgebra: Diagonal
using LocalProjections: kron_fastl, kron_fastr, getscore,
    ols, reg, _makeYX, _lp

function exampledata(name::Union{Symbol,String})
    path = (@__DIR__)*"/../data/$(name).csv.gz"
    # Need to allow CSV v0.8 to work
    f = open(path) |> GzipDecompressorStream |> read |> CSV.File
    return DataFrame(f, copycols=true)
end

const tests = [
    "utils",
    "lp",
    "irf"
]

printstyled("Running tests:\n", color=:blue, bold=true)

@time for test in tests
    include("$test.jl")
    println("\033[1m\033[32mPASSED\033[0m: $(test)")
end
