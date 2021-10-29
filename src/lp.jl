struct OLS{TF<:AbstractFloat} <: RegressionModel
    x::Matrix{TF}
    invxx::Matrix{TF}
    coef::VecOrMat{TF}
    resid::VecOrMat{TF}
    score::Matrix{TF}
end

function ols(Y::AbstractMatrix, X::AbstractMatrix)
    x = convert(Matrix, X)
    crossx = cholesky!(x'x)
    coef = x'Y
    ldiv!(crossx, coef)
    invxx = inv!(crossx)
    resid = Y - x * coef
    score = getscore(x, resid)
    return OLS(x, invxx, coef, resid, score)
end

modelmatrix(m::OLS) = m.x
coef(m::OLS) = m.coef
residuals(m::OLS) = m.resid

reg(Y::AbstractMatrix, X::AbstractMatrix, vce::Nothing) = (coef(ols(Y, X)), nothing)

function reg(Y::AbstractMatrix, X::AbstractMatrix, vce::CovarianceEstimator)
    m = ols(Y, X)
    return coef(m), vcov(m, vce)
end

"""
    AbstractEstimator

Supertype for all estimators.
"""
abstract type AbstractEstimator end

@fieldequal AbstractEstimator

show(io::IO, e::AbstractEstimator) = print(io, typeof(e).name.name)

"""
    LeastSquareLP <: AbstractEstimator

Ordinary least square estimator for local projecitons.

# Reference
Jordà, Òscar. 2005. "Estimation and Inference of Impulse Responses by Local Projections."
American Economic Review 95 (1): 161-182.
"""
struct LeastSquareLP <: AbstractEstimator end

show(io::IO, ::MIME"text/plain", ::LeastSquareLP) =
    print(io, "Ordinary Least Square Local Projection")

abstract type AbstractEstimatorResult end

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
        VCE<:CovarianceEstimator, TF<:AbstractFloat} <: StatisticalModel
    B::Array{TF,3}
    V::Array{TF,3}
    T::Vector{Int}
    est::TE
    estres::ER
    vce::VCE
    ynames::Vector{VarName}
    xnames::Vector{VarName}
    wnames::Vector{VarName}
    lookupy::Dict{VarName,Int}
    lookupx::Dict{VarName,Int}
    lookupw::Dict{VarName,Int}
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

function coef(r::LocalProjectionResult, horz::Int, xwname::VarName;
        yname::VarName=r.ynames[1], lag::Int=0)
    if lag > 0
        k = length(r.xnames)+r.lookupw[xwname]+(lag-1)*length(r.wnames)
        return r.B[k, r.lookupy[yname], horz]
    else
        return r.B[r.lookupx[xwname], r.lookupy[yname], horz]
    end
end

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

_toint(data, name::Symbol) = Tables.columnindex(data, name)
_toint(data, i::Integer) = Int(i)
_toint(data, c::Cum) = Cum(_toint(data, _geto(c)))

_toname(data, name::Symbol) = name
_toname(data, i::Integer) = Tables.columnnames(data)[i]
_toname(data, c::Cum) = Cum(_toname(data, _geto(c)))

function _getcols(data, ynames, xnames, wnames, addylag, nocons; TF=Float64)
    # The names must be iterable
    ynames isa VarIndex && (ynames = (ynames,))
    xnames isa VarIndex && (xnames = (xnames,))
    wnames isa VarIndex && (wnames = (wnames,))
    argmsg = ", must contain either integers, `Symbol`s or `TransformedVar`s"
    _checknames(ynames) || throw(ArgumentError("invalid ynames"*argmsg))
    length(xnames)==0 || _checknames(xnames) || throw(ArgumentError("invalid xnames"*argmsg))
    length(wnames)==0 || _checknames(wnames) || throw(ArgumentError("invalid wnames"*argmsg))

    # Convert all column indices to Int
    ynames = VarIndex[_toint(data, n) for n in ynames]
    xnames = VarIndex[_toint(data, n) for n in xnames]
    wnames = VarIndex[_toint(data, n) for n in wnames]
    addylag && union!(wnames, ynames)

    ys = Any[getcolumn(data, n) for n in ynames]
    xs = Any[getcolumn(data, n) for n in xnames]
    ws = Any[getcolumn(data, n) for n in wnames]
    if !nocons
        push!(xs, ones(TF, length(ys[1])))
        xnames = (xnames..., :constant)
    end
    return ynames, xnames, wnames, ys, xs, ws
end

function _makeYX(ys, xs, ws, nlag::Int, horz::Int, subset::Union{BitVector,Nothing};
        firststage::Bool=false, TF=Float64)
    ny = length(ys)
    ny > 0 || throw(ArgumentError("ys cannot be empty"))
    Tfull = size(ys[1],1)
    nlag > 0 || throw(ArgumentError("nlag must be at least 1"))
    # Number of rows possibly involved in estimation
    T = Tfull - nlag - horz
    nx = length(xs)
    nw = length(ws)
    nw > 0 || throw(ArgumentError("ws cannot be empty"))
    T > nx+nw*nlag || throw(ArgumentError(
        "not enough observations for nlag=$nlag and horz=$(horz)"))
    # Indicators for valid observations within the T rows
    esampleT = trues(T)
    # A cache for indicators
    aux = BitVector(undef, T)
    # Construct matrices
    Y = Matrix{TF}(undef, T, ny)
    for j in 1:ny
        if firststage
            v = view(vec(ys[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
            subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull-horz))
        else
            v = view(vec(ys[j], subset, :y, horz, TF), nlag+horz+1:Tfull)
            subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+horz+1:Tfull))
        end
        _esample!(esampleT, aux, v)
        copyto!(view(Y,esampleT,j), view(v,esampleT))
    end
    X = Matrix{TF}(undef, T, nx+nw*nlag)
    if nx > 0
        for j in 1:nx
            v = view(vec(xs[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
            subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull-horz))
            _esample!(esampleT, aux, v)
            copyto!(view(X,esampleT,j), view(v,esampleT))
        end
    end
    for j in 1:nw
        for l in 1:nlag
            v = view(vec(ws[j], subset, :x, horz, TF), nlag+1-l:Tfull-horz-l)
            subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1-l:Tfull-horz-l))
            _esample!(esampleT, aux, v)
            # Variables with the same lag are put together
            copyto!(view(X,esampleT,nx+(l-1)*nw+j), view(v,esampleT))
        end
    end
    T1 = sum(esampleT)
    if T1 < T
        T1 > size(X,2) || throw(ArgumentError(
            "not enough observations for nlag=$nlag and horz=$(horz)"))
        T = T1
        Y = Y[esampleT, :]
        X = X[esampleT, :]
    end
    return Y, X, T, esampleT
end

function _lp(ys, xs, ws, nlag::Int, horz::Int, vce::Union{CovarianceEstimator,Nothing},
        subset::Union{BitVector,Nothing}; TF=Float64)
    Y, X, T, esampleT = _makeYX(ys, xs, ws, nlag, horz, subset; TF=TF)
    return reg(Y, X, vce)..., T
end

function _normalize!(data, normalize, xnames, xs, ws, nlag, minhorz, subset; TF=Float64)
    snames = normalize isa Pair ? (normalize[1],) : (p[1] for p in normalize)
    ix_s = Vector{Int}(undef, length(snames))
    for (i, n) in enumerate(snames)
        fr = findfirst(x->x==_toint(data, n), xnames)
        fr === nothing && throw(ArgumentError(
            "Regressor $(n) is not found among x variables"))
        ix_s[i] = fr
    end
    nnames = normalize isa Pair ? (normalize[2],) : (p[2] for p in normalize)
    yn = Any[getcolumn(data, n) for n in nnames]
    Bn, Vn, Tn = _lp(yn, xs, ws, nlag, minhorz, nothing, subset; TF=TF)
    normmults = TF[Bn[ix,i] for (i,ix) in enumerate(ix_s)]
    xs[ix_s] .*= normmults
    normnames = VarName[_toname(data, n) for n in snames]
    normtars = VarName[_toname(data, n) for n in nnames]
    return normnames, normtars, normmults
end

_normalize!(data, normalize::Nothing, xnames, xs, ws, nlag, minhorz, subset; TF=Float64) =
    (nothing, nothing, nothing)

function _firststage(ys, xs, ws, nlag::Int, horz::Int, subset::Union{BitVector,Nothing};
        TF=Float64)
    Y, X, T, esampleT = _makeYX(ys, xs, ws, nlag, horz, subset; firststage=true, TF=TF)
    # Get first-stage estimates
    bf = X'Y
    ldiv!(cholesky!(X'X), bf)
    # Make the fitted values, need to have the same length as original data
    Tfull = size(ys[1],1)
    nY = length(ys)
    fitted = fill(fill(convert(TF, NaN), Tfull), nY)
    Xb = X * bf
    for i in 1:nY
        copyto!(view(view(fitted[i], nlag+1:Tfull-horz), esampleT), view(Xb,:,i))
    end
    return fitted
end

function _iv!(data, iv, firststagebyhorz, xnames, xs, ws, nlag, minhorz, subset; TF=Float64)
    endonames = iv[1]
    endonames isa VarIndex && (endonames = (endonames,))
    ix_iv = Vector{Int}(undef, length(endonames))
    for (i, n) in enumerate(endonames)
        fr = findfirst(x->x==_toint(data, n), xnames)
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
        xs[ix_iv] .= _firststage(yfs, xfs, ws, nlag, minhorz, subset; TF=TF)
    end
    endonames = VarName[_toname(data, n) for n in endonames]
    ivnames = VarName[_toname(data, n) for n in ivnames]
    return endonames, ivnames, ix_iv, yfs, xfs
end

_iv!(data, iv::Nothing, firststagebyhorz, xnames, xs, ws, nlag, minhorz, subset; TF=Float64) =
    (nothing, nothing, nothing, nothing, nothing)

function _est(::LeastSquareLP, data, xnames, ys, xs, ws, nlag, minhorz, nhorz, vce, subset,
        iv, ix_iv, yfs, xfs, firststagebyhorz; TF=Float64)
    ny = length(ys)
    nr = length(xs) + length(ws)*nlag
    B = Array{TF,3}(undef, nr, ny, nhorz)
    V = Array{TF,3}(undef, nr*ny, nr*ny, nhorz)
    T = Vector{Int}(undef, nhorz)
    for h in minhorz:minhorz+nhorz-1
        if iv !== nothing && firststagebyhorz
            xs[ix_iv] .= _firststage(yfs, xfs, ws, nlag, h, subset; TF=TF)
        end
        i = h - minhorz + 1
        Bh, Vh, T[i] = _lp(ys, xs, ws, nlag, h, vce, subset; TF=TF)
        B[:,:,i] = reshape(Bh, nr, ny, 1)
        V[:,:,i] = reshape(Vh, nr*ny, nr*ny, 1)
    end
    return B, V, T, nothing
end

lp(data, ynames; kwargs...) = lp(LeastSquareLP(), data, ynames; kwargs...)

function lp(estimator, data, ynames;
        xnames=(), wnames=(), nlag::Int=1, nhorz::Int=1, minhorz::Int=0,
        normalize::Union{VarIndexPair,Vector{VarIndexPair},Nothing}=nothing,
        iv::Union{Pair,Nothing}=nothing, firststagebyhorz::Bool=false,
        vce::CovarianceEstimator=HRVCE(),
        subset::Union{BitVector,Nothing}=nothing,
        addylag::Bool=true, nocons::Bool=false, TF::Type=Float64)

    checktable(data)
    ynames, xnames, wnames, ys, xs, ws =
        _getcols(data, ynames, xnames, wnames, addylag, nocons)

    normalize !== nothing && iv !== nothing &&
        throw(ArgumentError("options normalize and iv cannot be specified at the same time"))
    normnames, normtars, normmults =
        _normalize!(data, normalize, xnames, xs, ws, nlag, minhorz, subset, TF=TF)
    endonames, ivnames, ix_iv, yfs, xfs =
        _iv!(data, iv, firststagebyhorz, xnames, xs, ws, nlag, minhorz, subset, TF=TF)

    B, V, T, er = _est(estimator, data, xnames, ys, xs, ws, nlag, minhorz, nhorz, vce, subset,
        iv, ix_iv, yfs, xfs, firststagebyhorz, TF=TF)

    ynames = VarName[_toname(data, i) for i in ynames]
    xnames = VarName[_toname(data, i) for i in xnames]
    wnames = VarName[_toname(data, i) for i in wnames]
    return LocalProjectionResult(B, V, T, estimator, er, vce, ynames, xnames, wnames,
        Dict{VarName,Int}(n=>i for (i,n) in enumerate(ynames)),
        Dict{VarName,Int}(n=>i for (i,n) in enumerate(xnames)),
        Dict{VarName,Int}(n=>i for (i,n) in enumerate(wnames)),
        nlag, minhorz, subset, normnames, normtars, normmults,
        endonames, ivnames, firststagebyhorz, nocons)
end

show(io::IO, r::LocalProjectionResult) = print(io, typeof(r).name.name)

_vartitle(r::LocalProjectionResult) = "Variable Specifications"

function _varinfo(r::LocalProjectionResult, halfwidth::Int)
    namex = "Regressor"*(length(r.xnames) > 1 ? "s" : "")
    xs = join(string.(r.xnames), " ")
    namex = length(namex) + length(xs) > halfwidth ? (namex,) : namex
    namew = "Lagged control"*(length(r.wnames) > 1 ? "s" : "")
    ws = join(string.(r.wnames), " ")
    namew = length(namew) + length(ws) > halfwidth ? (namew,) : namew
    info = Pair[
        "Outcome variable"*(length(r.ynames) > 1 ? "s" : "") => join(string.(r.ynames), " "),
        "Minimum horizon" => string(r.minhorz),
        namex => xs,
        namew => ws]
    if r.endonames !== nothing
        push!(info, "Endogenous variable"*(length(r.endonames) > 1 ? "s" : "") =>
            join(string.(r.endonames), " "),
            "Instrument"*(length(r.ivnames) > 1 ? "s" : "") =>
            join(string.(r.ivnames), " "))
    end
    return info
end

_estimatortitle(::LocalProjectionResult) = nothing
_estimatorinfo(::LocalProjectionResult, ::Int) = nothing

function show(io::IO, ::MIME"text/plain", r::LocalProjectionResult;
        totalwidth::Int=80, interwidth::Int=4+mod(totalwidth,2))
    H = length(r.T)
    print(io, "$(typeof(r).name.name) with $(r.nlag) lag")
    print(io, r.nlag > 1 ? "s " : " ", "over $H horizon")
    println(io, H > 1 ? "s:" : ":")
    halfwidth = div(totalwidth-interwidth, 2)
    titles = (_vartitle(r), _estimatortitle(r))
    blocks = (_varinfo(r, halfwidth), _estimatorinfo(r, halfwidth))
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
