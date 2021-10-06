# Convert the original example datasets to a compressed CSV file

# See data/README.md for the source of the input data files
# To regenerate the output file:
# 1) Have all input files ready in the data folder
# 2) Instantiate the package environment for data/src
# 3) Run this script with the root folder of the repository being the working directory

using CSV, CodecZlib, DataFrames

function gk()
    var = CSV.read("data/VAR_data.csv", DataFrame)
    iv = CSV.read("data/factor_data.csv", DataFrame)
    # Merge the two datasets
    out = innerjoin(var, iv, on=[:year, :month])
    open(GzipCompressorStream, "data/gk.csv.gz", "w") do stream
        CSV.write(stream, out)
    end
end

function hp()
    hp = CSV.read("data/employment.csv", DataFrame, header=[:quarter, :payrolls_a, :payrolls_u])
    open(GzipCompressorStream, "data/hp.csv.gz", "w") do stream
        CSV.write(stream, hp)
    end
end

function main()
    gk()
    hp()
end

main()
