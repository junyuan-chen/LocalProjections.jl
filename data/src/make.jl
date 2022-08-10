# Convert the original example datasets to a compressed CSV file

# See data/README.md for the source of the input data files
# To regenerate the output file:
# 1) Have all input files ready in the data folder
# 2) Instantiate the package environment for data/src
# 3) Run this script with the root folder of the repository being the working directory

using CSV, CodecZlib, DataFrames, ReadStatTables, ShiftedArrays, XLSX

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
    # Add variables needed for state dependent results
    # Construct the lag of slack
    df.rec = similar(df.slack)
    df.rec[1] = missing
    # Fill the beginning missing values
    df.rec[2:5] .= true
    df.rec[6:end] .= view(df.slack, 5:length(df.rec)-1)
    df.exp = 1.0 .- df.rec
    df.recnewsy = df.rec.*df.newsy
    df.expnewsy = df.exp.*df.newsy
    df.recg = df.rec.*df.g
    df.expg = df.exp.*df.g
    open(GzipCompressorStream, "data/rz.csv.gz", "w") do stream
        CSV.write(stream, df)
    end
end

function bb()
    df = CSV.read("data/data.csv", DataFrame)
    open(GzipCompressorStream, "data/bb.csv.gz", "w") do stream
        CSV.write(stream, df)
    end
end

lagdiff(x) = x - lag(x)

function jst()
    cols = [:iso, :year, :rgdpmad, :pop, :cpi, :stir]
    df = DataFrame(readstat("data/JSTdatasetR6.dta", usecols=cols))
    df[!,:lgrgdp] = 100.0.*log.(df.rgdpmad.*df.pop)
    df[!,:lgcpi] = 100.0.*log.(df.cpi)
    df[!,:year] = convert(Vector{Int}, df.year)
    df = df[!,Not([:rgdpmad,:pop])]
    transform!(groupby(df, :iso),
        :lgrgdp=>lagdiff=>:dlgrgdp, :lgcpi=>lagdiff=>:dlgcpi, :stir=>lagdiff=>:dstir)
    open(GzipCompressorStream, "data/jst.csv.gz", "w") do stream
        CSV.write(stream, df)
    end
end

function main()
    gk()
    hp()
    rz()
    bb()
    jst()
end

main()
