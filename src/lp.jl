"""
    OLS{TF<:AbstractFloat} <: RegressionModel

Data from an ordinary least squares regression.
"""
struct OLS{TF<:AbstractFloat} <: RegressionModel
    X::Matrix{TF}
    invXX::Matrix{TF}
    coef::VecOrMat{TF}
    resid::VecOrMat{TF}
    score::Matrix{TF}
end

function OLS(Y::AbstractVecOrMat, X::AbstractMatrix)
    X = convert(Matrix, X)
    crossx = cholesky!(X'X)
    coef = X'Y
    ldiv!(crossx, coef)
    invXX = inv!(crossx)
    resid = X * coef
    resid .= Y .- resid
    score = getscore(X, resid)
    return OLS(X, invXX, coef, resid, score)
end

modelmatrix(m::OLS) = m.X
coef(m::OLS) = m.coef
residuals(m::OLS) = m.resid

show(io::IO, ::OLS) = print(io, "OLS regression")

"""
    AbstractEstimator

Supertype for all estimators.
"""
abstract type AbstractEstimator end

@fieldequal AbstractEstimator

show(io::IO, e::AbstractEstimator) = print(io, typeof(e).name.name)

"""
    LeastSquaresLP <: AbstractEstimator

Ordinary least squares estimator for local projecitons.

# Reference
Jordà, Òscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections."
American Economic Review 95 (1): 161-182.
"""
struct LeastSquaresLP <: AbstractEstimator end

show(io::IO, ::MIME"text/plain", ::LeastSquaresLP) =
    print(io, "Ordinary least squares local projection")

"""
    AbstractEstimatorResult

Supertype for all estimator-specific additional results.
"""
abstract type AbstractEstimatorResult end

show(io::IO, er::AbstractEstimatorResult) = print(io, typeof(er).name.name)

"""
    LeastSquaresLPResult{TF<:AbstractFloat} <: AbstractEstimatorResult

Additional results from estimating least squares local projections.
See also [`LeastSquaresLP`](@ref) and [`LocalProjectionResult`](@ref).

# Field
- `ms::Vector{OLS{TF}}`: data from the OLS regressions.
"""
struct LeastSquaresLPResult{TF<:AbstractFloat} <: AbstractEstimatorResult
    ms::Vector{OLS{TF}}
end

"""
    VarName

A type union of all accepted types for indexing variables by name.
"""
const VarName = Union{Symbol,TransformedVar{Symbol}}

"""
    LocalProjectionResult

Results from local projection estimation.
"""
struct LocalProjectionResult{TE<:AbstractEstimator,
        ER<:Union{AbstractEstimatorResult,Nothing},
        VCE<:CovarianceEstimator, TP<:Union{Symbol,Nothing},
        TF<:AbstractFloat} <: StatisticalModel
    B::Array{TF,3}
    V::Array{TF,3}
    T::Vector{Int}
    est::TE
    estres::ER
    vce::VCE
    ynames::Vector{VarName}
    xnames::Vector{VarName}
    wnames::Vector{VarName}
    stnames::Vector{VarName}
    fenames::Vector{VarName}
    lookupy::Dict{VarName,Int}
    lookupx::Dict{VarName,Int}
    lookupw::Dict{VarName,Int}
    panelid::TP
    panelweight::Union{Symbol,Nothing}
    nlag::Int
    minhorz::Int
    subset::Union{BitVector,Nothing}
    normnames::Union{Vector{VarName},Nothing}
    normtars::Union{Vector{VarName},Nothing}
    normmults::Union{Vector{TF},Nothing}
    endonames::Union{Vector{VarName},Nothing}
    ivnames::Union{Vector{VarName},Nothing}
    firststagebyhorz::Bool
    nocons::Bool
end

"""
    coef(r::LocalProjectionResult, horz::Int, xwname::VarName; kwargs...)

Return the coefficient estimate at horizon `horz` for variable with column name `xwname`.

# Keywords
- `yname::VarName=r.ynames[1]`: name of the outcome variable.
- `lag::Int=0`: lag of variable `xwname`; being 0 means the variable is contemporaneous.
"""
function coef(r::LocalProjectionResult, horz::Int, xwname::VarName;
        yname::VarName=r.ynames[1], lag::Int=0)
    if lag > 0
        k = length(r.xnames)+r.lookupw[xwname]+(lag-1)*length(r.wnames)
        return r.B[k, r.lookupy[yname], horz]
    else
        return r.B[r.lookupx[xwname], r.lookupy[yname], horz]
    end
end

"""
    vcov(r::LocalProjectionResult, horz::Int, xwname1::VarName; kwargs...)

Return the variance of the estimate at horizon `horz` for variable with column name `xwname1`.
If `xwname2` is specified, return the covariance between `xwname1` and `xwname2`.

# Keywords
- `yname1::VarName=r.ynames[1]`: name of the outcome variable for `xwname1`.
- `lag1::Int=0`: lag of variable `xwname1`; being 0 means the variable is contemporaneous.
- `xwname2::VarName=xwname1`: the second variable involved in the covariance.
- `yname2::VarName=yname1`: name of the outcome variable for `xwname1`.
- `lag2::Int=lag1`: lag of variable `xwname2`; being 0 means the variable is contemporaneous.
"""
function vcov(r::LocalProjectionResult, horz::Int, xwname1::VarName;
        yname1::VarName=r.ynames[1], lag1::Int=0, xwname2::VarName=xwname1,
        yname2::VarName=yname1, lag2::Int=lag1)
    if lag1 > 0
        k1 = length(r.xnames)+r.lookupw[xwname1]+(lag1-1)*length(r.wnames)
    else
        k1 = r.lookupx[xwname1]
    end
    if lag2 > 0
        k2 = length(r.xnames)+r.lookupw[xwname2]+(lag2-1)*length(r.wnames)
    else
        k2 = r.lookupx[xwname2]
    end
    n1 = r.lookupy[yname1]
    n2 = r.lookupy[yname2]
    return r.V[(k1-1)*length(r.ynames)+n1, (k2-1)*length(r.ynames)+n2, horz]
end

"""
    VarIndex

A type union of all accepted types for indexing variables.
"""
const VarIndex = Union{Integer, Symbol, TransformedVar{<:Union{Integer,Symbol}}}

"""
    VarIndexPair

`Pair` type with both type parameters being a subtype of [`VarIndex`](@ref).
"""
const VarIndexPair = Pair{<:VarIndex,<:VarIndex}

_checknames(names) = all(n isa VarIndex for n in names)

_toname(data, name::Symbol) = name
_toname(data, i::Integer) = Tables.columnnames(data)[i]

function _getcols(data, ynames, xnames, wnames, states, panelid, fes, panelweight,
        addpanelidfe, addylag, nocons; TF=Float64)
    # The names must be iterable
    ynames isa VarIndex && (ynames = (ynames,))
    xnames isa VarIndex && (xnames = (xnames,))
    wnames isa VarIndex && (wnames = (wnames,))
    states isa VarIndex && (states = (states,))
    fes isa VarIndex && (fes = (fes,))
    argmsg = ", must contain either integers, `Symbol`s or `TransformedVar`s"
    _checknames(ynames) || throw(ArgumentError("invalid ynames"*argmsg))
    length(xnames)==0 || _checknames(xnames) || throw(ArgumentError("invalid xnames"*argmsg))
    length(wnames)==0 || _checknames(wnames) || throw(ArgumentError("invalid wnames"*argmsg))
    states===nothing || _checknames(states) || throw(ArgumentError("invalid states"*argmsg))
    length(fes)==0 || _checknames(fes) || throw(ArgumentError("invalid fes"*argmsg))

    # Convert all column indices to Symbol
    ynames = VarName[_toname(data, n) for n in ynames]
    xnames = VarName[_toname(data, n) for n in xnames]
    wnames = VarName[_toname(data, n) for n in wnames]
    stnames = states === nothing ? VarName[] : VarName[_toname(data, n) for n in states]
    fenames = VarName[_toname(data, n) for n in fes]
    panelid === nothing || (panelid = _toname(data, panelid))
    panelweight === nothing || (panelweight = _toname(data, panelweight))
    !addpanelidfe || panelid === nothing || panelid in fenames || push!(fenames, panelid)
    addylag && union!(wnames, ynames)

    ys = Any[getcolumn(data, n) for n in ynames]
    xs = Any[getcolumn(data, n) for n in xnames]
    ws = Any[getcolumn(data, n) for n in wnames]
    sts = states === nothing ? nothing : Any[getcolumn(data, n) for n in states]
    fes = Any[getcolumn(data, n) for n in fenames]
    if !nocons && isempty(fenames)
        push!(xs, ones(TF, length(ys[1])))
        xnames = VarName[xnames..., :constant]
    end

    groups = panelid === nothing ? nothing : _group(getcolumn(data, panelid))
    pw = panelweight === nothing ? nothing : getcolumn(data, panelweight)

    return ynames, xnames, wnames, stnames, fenames, panelid, panelweight, states,
        ys, xs, ws, sts, fes, pw, groups
end

function _lp(dt::LPData, horz::Int, vce::Union{CovarianceEstimator,Nothing}; TF=Float64)
    Y, X, W, T, esampleT = _makeYX(dt, horz)
    m = OLS(Y, X)
    return coef(m), vcov(m, vce), T, m
end

function _normalize!(data, normalize, xnames, xs, ws, sts, fes, pw, nlag, minhorz,
        subset, groups; TF=Float64)
    snames = normalize isa Pair ? (normalize[1],) : (p[1] for p in normalize)
    ix_s = Vector{Int}(undef, length(snames))
    for (i, n) in enumerate(snames)
        fr = findfirst(x->x==_toname(data, n), xnames)
        fr === nothing && throw(ArgumentError(
            "Regressor $(n) is not found among x variables"))
        ix_s[i] = fr
    end
    nnames = normalize isa Pair ? (normalize[2],) : (p[2] for p in normalize)
    yn = Any[getcolumn(data, n) for n in nnames]
    dt = LPData(yn, xs, ws, sts, fes, pw, nlag, minhorz, subset, groups, TF)
    Bn, Vn, Tn, _ = _lp(dt, minhorz, nothing)
    normmults = TF[Bn[ix,i] for (i,ix) in enumerate(ix_s)]
    xs[ix_s] .*= normmults
    normnames = VarName[_toname(data, n) for n in snames]
    normtars = VarName[_toname(data, n) for n in nnames]
    return normnames, normtars, normmults
end

_normalize!(data, normalize::Nothing, xnames, xs, ws, sts, fes, pw, nlag, minhorz, subset,
    groups; TF=Float64) = (nothing, nothing, nothing)

function _firststage(ys, xs, ws, sts, fes, pw, nlag::Int, horz::Int,
        subset::Union{BitVector,Nothing}, groups; TF=Float64)
    dt = LPData(ys, xs, ws, sts, fes, pw, nlag, horz, subset, groups, TF)
    Y, X, W, T, esampleT = _makeYX(dt, horz, true)
    # Get first-stage estimates
    bf = X'Y
    ldiv!(cholesky!(X'X), bf)
    # Make the fitted values, need to have the same length as original data
    Tfull = size(ys[1],1)
    nY = length(ys)
    fitted = ((fill(convert(TF, NaN), Tfull) for _ in 1:nY)...,)
    Xb = X * bf
    # Need to rescale back to avoid multiplying weights twice
    pw === nothing || (Xb ./= sqrt.(W))
    for i in 1:nY
        copyto!(view(view(fitted[i], nlag+1:Tfull-horz), esampleT), view(Xb,:,i))
    end
    return fitted
end

function _iv!(data, iv, firststagebyhorz, xnames, xs, ws, sts, fes, pw, nlag, minhorz, subset,
        groups; TF=Float64)
    endonames = iv[1]
    endonames isa VarIndex && (endonames = (endonames,))
    ix_iv = Vector{Int}(undef, length(endonames))
    for (i, n) in enumerate(endonames)
        fr = findfirst(x->x==_toname(data, n), xnames)
        fr === nothing && throw(ArgumentError(
            "Endogenous variable $(n) is not found among x variables"))
        ix_iv[i] = fr
    end
    ivnames = iv[2]
    ivnames isa VarIndex && (ivnames = (ivnames,))
    length(ivnames)==0 && throw(ArgumentError("invalid specification of option iv"))
    argmsg = ", must contain either integers, `Symbol`s or `TransformedVar`s"
    _checknames(ivnames) || throw(ArgumentError("invalid ivnames"*argmsg))
    # Collect columns used for first-stage regression
    yfs = xs[ix_iv]
    xfs = Any[(xs[i] for i in 1:length(xs) if !(i in ix_iv))...,
        (getcolumn(data, n) for n in ivnames)...]
    if !firststagebyhorz
        any(x->x isa Cum, endonames) &&
            @warn "firststagebyhorz=false while endogenous variables contain Cum"
        # Replace the endogenous variable with the fitted values
        xs[ix_iv] .= _firststage(yfs, xfs, ws, sts, fes, pw, nlag, minhorz, subset, groups;
            TF=TF)
    end
    endonames = VarName[_toname(data, n) for n in endonames]
    ivnames = VarName[_toname(data, n) for n in ivnames]
    return endonames, ivnames, ix_iv, yfs, xfs
end

_iv!(data, iv::Nothing, firststagebyhorz, xnames, xs, ws, sts, fes, pw, nlag, minhorz, subset,
    groups; TF=Float64) = (nothing, nothing, nothing, nothing, nothing)

function _est(::LeastSquaresLP, data, xnames, ys, xs, ws, sts, fes, pw,
        nlag, minhorz, nhorz, vce, subset, groups,
        iv, ix_iv, yfs, xfs, firststagebyhorz; TF=Float64)
    ny = length(ys)
    nstate =sts === nothing ? 1 : length(sts)
    nr = length(xs) + length(ws)*nlag*nstate
    B = Array{TF,3}(undef, nr, ny, nhorz)
    V = Array{TF,3}(undef, nr*ny, nr*ny, nhorz)
    T = Vector{Int}(undef, nhorz)
    M = Vector{OLS{TF}}(undef, nhorz)
    if !(iv !== nothing && firststagebyhorz) && !any(x->x isa Cum, xs)
        dt = LPData(ys, xs, ws, sts, fes, pw, nlag, minhorz, subset, groups, TF)
    end
    for h in minhorz:minhorz+nhorz-1
        if iv !== nothing && firststagebyhorz
            xs[ix_iv] .= _firststage(yfs, xfs, ws, sts, fes, pw, nlag, h, subset, groups;
                TF=TF)
            dt = LPData(ys, xs, ws, sts, fes, pw, nlag, minhorz, subset, groups, TF)
        elseif any(x->x isa Cum, xs)
            dt = LPData(ys, xs, ws, sts, fes, pw, nlag, minhorz, subset, groups, TF)
        end
        i = h - minhorz + 1
        Bh, Vh, T[i], M[i] = _lp(dt, h, vce)
        B[:,:,i] = reshape(Bh, nr, ny, 1)
        V[:,:,i] = reshape(Vh, nr*ny, nr*ny, 1)
    end
    return B, V, T, LeastSquaresLPResult(M)
end

"""
    lp([estimator], data, ynames; kwargs...)

Estimate local projections with the specified `estimator`
for outcome variable(s) with column name(s) `ynames` in `data` table.
If the `estimator` is not specified, [`LeastSquaresLP`](@ref) is assumed.
The input `data` must be `Tables.jl`-compatible.

# Keywords
- `xnames=()`: indices of contemporaneous regressors from `data`.
- `wnames=()`: indices of lagged control variables from `data`.
- `nlag::Int=4`: number of lags to be included for the lagged control variables.
- `nhorz::Int=1`: total number of horizons to be estimated.
- `minhorz::Int=0`: the minimum horizon involved in estimation.
- `normalize::Union{VarIndexPair,Vector{VarIndexPair},Nothing}=nothing`: normalize the magnitude of the specified regressor(s) based on their initial impact on the respective targeted outcome variable.
- `iv::Union{Pair,Nothing}=nothing`: endogenous varable(s) paired with instrument(s).
- `firststagebyhorz::Bool=false`: estimate the first-stage regression separately for each horizon.
- `states=nothing`: variable(s) representing the states for a state-dependent specification.
- `panelid::Union{Symbol,Integer,Nothing}=nothing`: variable identifying the units in a panel.
- `fes=()`: variable(s) identifying the fixed effects.
- `addpanelidfe::Bool=true`: use `panelid` for unit fixed effects.
- `panelweight::Union{Symbol,Integer,Nothing}=nothing`: weights across units in a panel.
- `vce::CovarianceEstimator=HARVCE(EWC())`: the variance-covariance estimator.
- `subset::Union{BitVector,Nothing}=nothing`: subset of `data` to be used for estimation.
- `addylag::Bool=true`: include lags of the outcome variable(s).
- `nocons::Bool=false`: do not add the constant term.
- `TF::Type=Float64`: numeric type used for estimation.
"""
function lp(estimator, data, ynames;
        xnames=(), wnames=(), nlag::Int=4, nhorz::Int=1, minhorz::Int=0,
        normalize::Union{VarIndexPair,Vector{VarIndexPair},Nothing}=nothing,
        iv::Union{Pair,Nothing}=nothing, firststagebyhorz::Bool=false, states=nothing,
        panelid::Union{Symbol,Integer,Nothing}=nothing, fes=(), addpanelidfe::Bool=true,
        panelweight::Union{Symbol,Integer,Nothing}=nothing,
        vce::CovarianceEstimator=HARVCE(EWC()),
        subset::Union{BitVector,Nothing}=nothing,
        addylag::Bool=true, nocons::Bool=false, TF::Type=Float64)

    checktable(data)
    if panelweight !== nothing && panelid === nothing
        @warn "panelweight is ignored when panelid is nothing"
        panelweight = nothing
    end
    ynames, xnames, wnames, stnames, fenames, panelid, panelweight, states, ys, xs, ws,
        sts, fes, pw, groups = _getcols(data, ynames, xnames, wnames, states, panelid,
            fes, panelweight, addpanelidfe, addylag, nocons)

    if any(x->x isa Cum, ynames)
        addylag && @warn "addylag=true while outcome variables contain Cum"
    end

    normalize !== nothing && iv !== nothing &&
        throw(ArgumentError("options normalize and iv cannot be specified at the same time"))
    normnames, normtars, normmults =
        _normalize!(data, normalize, xnames, xs, ws, sts, fes, pw, nlag, minhorz, subset,
            groups, TF=TF)
    endonames, ivnames, ix_iv, yfs, xfs =
        _iv!(data, iv, firststagebyhorz, xnames, xs, ws, sts, fes, pw, nlag, minhorz, subset,
            groups, TF=TF)

    B, V, T, er = _est(estimator, data, xnames, ys, xs, ws, sts, fes, pw, nlag, minhorz, nhorz,
        vce, subset, groups, iv, ix_iv, yfs, xfs, firststagebyhorz, TF=TF)

    return LocalProjectionResult(B, V, T, estimator, er, vce, ynames, xnames, wnames,
        stnames, fenames,
        Dict{VarName,Int}(n=>i for (i,n) in enumerate(ynames)),
        Dict{VarName,Int}(n=>i for (i,n) in enumerate(xnames)),
        Dict{VarName,Int}(n=>i for (i,n) in enumerate(wnames)),
        panelid, panelweight, nlag, minhorz, subset, normnames, normtars, normmults,
        endonames, ivnames, firststagebyhorz, nocons)
end

lp(data, ynames; kwargs...) = lp(LeastSquaresLP(), data, ynames; kwargs...)

"""
    lp(r::LocalProjectionResult, vce::CovarianceEstimator)

Reestimate the variance-covariance matrices with `vce`.
"""
function lp(r::LocalProjectionResult{LeastSquaresLP}, vce::CovarianceEstimator)
    V = similar(r.V)
    K = size(V, 1)
    for h in 1:length(r.T)
        V[:,:,h] = reshape(vcov(r.estres.ms[h], vce), K, K, 1)
    end
    return LocalProjectionResult(r.B, V, r.T, r.est, r.estres, vce,
        r.ynames, r.xnames, r.wnames, r.stnames, r.fenames, r.lookupy, r.lookupx, r.lookupw,
        r.panelid, r.panelweight, r.nlag, r.minhorz, r.subset,
        r.normnames, r.normtars, r.normmults,
        r.endonames, r.ivnames, r.firststagebyhorz, r.nocons)
end

show(io::IO, r::LocalProjectionResult) = print(io, typeof(r).name.name)

_vartitle(::LocalProjectionResult) = "Variable Specifications"

function _varinfo(r::LocalProjectionResult, halfwidth::Int)
    namex = "Regressor"*(length(r.xnames) > 1 ? "s" : "")
    xs = join(r.xnames, " ")
    namex = length(namex) + length(xs) > halfwidth ? (namex,) : namex
    namew = "Lagged control"*(length(r.wnames) > 1 ? "s" : "")
    ws = join(r.wnames, " ")
    namew = length(namew) + length(ws) > halfwidth ? (namew,) : namew
    info = Pair[
        "Outcome variable"*(length(r.ynames) > 1 ? "s" : "") => join(r.ynames, " "),
        "Minimum horizon" => r.minhorz,
        namex => xs,
        namew => ws
    ]
    if r.endonames !== nothing
        push!(info, "Endogenous variable"*(length(r.endonames) > 1 ? "s" : "") =>
            join(r.endonames, " "),
            "Instrument"*(length(r.ivnames) > 1 ? "s" : "") =>
            join(r.ivnames, " "))
    end
    isempty(r.stnames) || push!(info, "States" => join(r.stnames, " "))
    return info
end

_paneltitle(::LocalProjectionResult{<:Any,<:Any,<:Any,<:Any}) = "Panel Specifications"
_paneltitle(::LocalProjectionResult{<:Any,<:Any,<:Any,Nothing}) = nothing

function _panelinfo(r::LocalProjectionResult{<:Any,<:Any,<:Any,<:Any}, halfwidth::Int)
    namefe = "Fixed effects"
    fes = join(r.fenames, " ")
    namefe = length(namefe) + length(fes) > halfwidth ? (namefe,) : namefe
    info = Pair[
        "Unit ID" => r.panelid,
        "Weight" => r.panelweight === nothing ? "(unit weight)" : r.panelweight,
        namefe => isempty(r.fenames) ? "(none)" : fes
    ]
    return info
end

_panelinfo(::LocalProjectionResult{<:Any,<:Any,<:Any,Nothing}, halfwidth::Int) =
    nothing

_estimatortitle(::LocalProjectionResult) = nothing
_estimatorinfo(::LocalProjectionResult, ::Int) = nothing

function show(io::IO, ::MIME"text/plain", r::LocalProjectionResult;
        totalwidth::Int=78, interwidth::Int=4+mod(totalwidth,2))
    H = length(r.T)
    print(io, "$(typeof(r).name.name) with $(r.nlag) lag")
    print(io, r.nlag > 1 ? "s " : " ", "over $H horizon")
    println(io, H > 1 ? "s:" : ":")
    halfwidth = div(totalwidth-interwidth, 2)
    titles = (_vartitle(r), _paneltitle(r), _estimatortitle(r))
    blocks = (_varinfo(r, halfwidth), _panelinfo(r, halfwidth), _estimatorinfo(r, halfwidth))
    for ib in 1:length(blocks)
        if blocks[ib] !== nothing
            println(io, repeat('─', totalwidth))
            if titles[ib] !== nothing
                println(io, titles[ib])
                println(io, repeat('─', totalwidth))
            end
            # Use count to determine whether a newline is needed
            count = 0
            for e in blocks[ib]
                # Allow the flexibility of printing on the entire row
                if e[1] isa Tuple
                    isodd(count) && print(io, '\n')
                    print(io, e[1][1], ':')
                    println(io, lpad(e[2], totalwidth - length(e[1][1]) - 1))
                    count = 0
                else
                    # Allow having vacancy by specifying an empty string
                    if e[1] != ""
                        print(io, e[1], ':')
                        print(io, lpad(e[2], halfwidth - length(e[1]) - 1))
                    end
                    count += 1
                    print(io, isodd(count) ? repeat(' ', interwidth) : '\n')
                end
            end
            isodd(count) && print(io, '\n')
        end
    end
    print(io, repeat('─', totalwidth))
end
