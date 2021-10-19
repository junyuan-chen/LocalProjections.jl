# Convert the original example datasets to a compressed CSV file

# See data/README.md for the source of the input data files
# To regenerate the output file:
# 1) Have all input files ready in the data folder
# 2) Instantiate the package environment for data/src
# 3) Run this script with the root folder of the repository being the working directory

using CSV, CodecZlib, DataFrames, ShiftedArrays, XLSX

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
    df = CSV.read("data/employment.csv", DataFrame, header=[:quarter, :payrolls_a, :payrolls_u])
    open(GzipCompressorStream, "data/hp.csv.gz", "w") do stream
        CSV.write(stream, df)
    end
end

function rz()
    df = DataFrame(XLSX.readtable("data/RZDAT.xlsx", "rzdat", infer_eltypes=true)...)
    df = df[df.quarter.>=1889,:]
    # Generate variables following the replication files
    df.wwii = (df.quarter.>=1941.5) .& (df.quarter.<1946)
    df.slack = df.unemp.>=6.5
    ynorm = df.rgdp_pott6
    df.newsy = df.news./(lag(ynorm).*lag(df.pgdp))
    df.rgov = df.ngov./df.pgdp
    df.y = df.rgdp./ynorm
    df.g = df.rgov./ynorm
    open(GzipCompressorStream, "data/rz.csv.gz", "w") do stream
        CSV.write(stream, df)
    end
end

function main()
    gk()
    hp()
    rz()
end

main()
