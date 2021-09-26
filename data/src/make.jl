# Convert the original example datasets to a compressed CSV file

# See data/README.md for the source of the input data files
# To regenerate the output file:
# 1) Have all input files ready in the data folder
# 2) Instantiate the package environment for data/src
# 3) Run this script with the root folder of the repository being the working directory

using CSV, CodecZlib, DataFrames

function main()
    var = CSV.read("data/VAR_data.csv", DataFrame)
    iv = CSV.read("data/factor_data.csv", DataFrame)
    # Merge the two datasets
    out = innerjoin(var, iv, on=[:year, :month])
    open(GzipCompressorStream, "data/gk.csv.gz", "w") do stream
        CSV.write(stream, out)
    end
end

main()
