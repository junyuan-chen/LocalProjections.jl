struct OLS{T<:AbstractFloat} <: RegressionModel
    x::Matrix{T}
    invxx::Matrix{T}
    coef::VecOrMat{T}
    resid::VecOrMat{T}
    score::Matrix{T}
end

function ols(Y::AbstractMatrix, X::AbstractMatrix)
    x = convert(Matrix, X)
    invxx = inv(x'x)
    coef = x \ Y
    resid = Y - x * coef
    score = getscore(x, resid)
    return OLS(x, invxx, coef, resid, score)
end

modelmatrix(m::OLS) = m.x
coef(m::OLS) = m.coef
residuals(m::OLS) = m.resid

function reg(Y::AbstractMatrix, X::AbstractMatrix, vce::CovarianceEstimator)
    m = ols(Y, X)
    return coef(m), vcov(m, vce)
end

struct LocalProjectionResult{TF<:AbstractFloat, VCE<:CovarianceEstimator} <: StatisticalModel
    B::Vector{Matrix{TF}}
    V::Vector{Matrix{TF}}
    T::Vector{Int}
    vce::VCE
    ynames::Vector{Symbol}
    xnames::Vector{Symbol}
    wnames::Vector{Symbol}
    lookupy::Dict{Symbol,Int}
    lookupxw::Dict{Symbol,Int}
    nlag::Int
    nhorz::Int
    nocons::Bool
end

coef(r::LocalProjectionResult, horz::Int, yname::Symbol, xwname::Symbol, lag::Int=1) =
    r.B[horz][r.lookupxw[xwname]+max((lag-1), 0)*length(r.wnames), r.lookupy[yname]]

function vcov(r::LocalProjectionResult, horz::Int, yname1::Symbol, xwname1::Symbol,
        lag1::Int=1, yname2::Symbol=yname1, xwname2::Symbol=xwname1, lag2::Int=lag1)
    k1 = r.lookupxw[xwname1]+max((lag1-1), 0)*length(r.wnames)
    n1 = r.lookupy[yname1]
    k2 = r.lookupxw[xwname2]+max((lag2-1), 0)*length(r.wnames)
    n2 = r.lookupy[yname2]
    return r.V[horz][(k1-1)*length(r.ynames)+n1, (k2-1)*length(r.ynames)+n2]
end

function _esample!(esample::BitVector, aux::BitVector, v::AbstractVector)
    aux .= isequal.(isfinite.(v), true)
    esample .&= aux
    if Missing <: eltype(v)
        aux .= .~(ismissing.(v))
        esample .&= aux
    end
end

function _makeYX(ys, xs, ws, nlag::Int, horz::Int; TF=Float64)
    ny = length(ys)
    ny > 0 || throw(ArgumentError("ys cannot be empty"))
    Tfull = size(ys[1],1)
    nlag > 0 || throw(ArgumentError("nlag must be at least 1"))
    # Number of rows involved in estimation
    T = Tfull - nlag - horz
    # Indicate whether a missing value exists in each row
    esample = trues(T)
    # A cache for indicators
    aux = BitVector(undef, T)
    Y = Matrix{TF}(undef, T, ny)
    for j in 1:ny
        size(ys[j],1) == Tfull ||
            throw(ArgumentError("incompatible length of the $(j)th outcome variable"))
        v = view(ys[j], nlag+horz+1:Tfull)
        _esample!(esample, aux, v)
        copyto!(view(Y,esample,j), view(v,esample))
    end
    nx = length(xs)
    nw = length(ws)
    nw > 0 || throw(ArgumentError("ws cannot be empty"))
    X = Matrix{TF}(undef, T, nx+nw*nlag)
    if nx > 0
        for j in 1:nx
            size(xs[j],1) == Tfull ||
                throw(ArgumentError("incompatible length of the $(j)th x variable"))
            v = view(xs[j], nlag+1:Tfull-horz)
            _esample!(esample, aux, v)
            copyto!(view(X,esample,j), view(v,esample))
        end
    end
    for j in 1:nw
        size(ws[j],1) == Tfull ||
            throw(ArgumentError("incompatible length of the $(j)th w variable"))
        for l in 1:nlag
            v = view(ws[j], nlag+1-l:Tfull-horz-l)
            _esample!(esample, aux, v)
            # Variables with the same lag are put together
            copyto!(view(X,esample,nx+(l-1)*nw+j), view(v,esample))
        end
    end
    T1 = sum(esample)
    if T1 < T
        T1 == 0 && throw(ArgumentError("no valid observation for nlag=$nlag and horz=$(horz)"))
        T = T1
        Y = Y[esample, :]
        X = X[esample, :]
    end
    return Y, X, T
end

function _lp(ys, xs, ws, nlag::Int, horz::Int, vce::CovarianceEstimator; TF=Float64)
    Y, X, T = _makeYX(ys, xs, ws, nlag, horz; TF=TF)
    return reg(Y, X, vce)..., T
end

_checknames(::Symbol) = true
_checknames(names) = all(n isa Union{Integer, Symbol} for n in names)

_toint(data, name::Symbol) = Tables.columnindex(data, name)
_toint(data, i::Integer) = Int(i)

_toname(data, name::Symbol) = name
_toname(data, i::Integer) = Tables.columnnames(data)[i]

function lp(data, ynames;
        xnames=(), wnames=(), nlag::Int=1, nhorz::Int=1,
        vce::CovarianceEstimator=HRVCE(),
        nocons::Bool=false, TF::Type=Float64)
    checktable(data)
    # The names must be iterable
    ynames isa Union{Symbol,Integer} && (ynames = (ynames,))
    xnames isa Union{Symbol,Integer} && (xnames = (xnames,))
    wnames isa Union{Symbol,Integer} && (wnames = (wnames,))
    _checknames(ynames) ||
        throw(ArgumentError("Invalid ynames, must contain either integers or `Symbol`s"))
    length(xnames)==0 || _checknames(xnames) ||
        throw(ArgumentError("Invalid xnames, must contain either integers or `Symbol`s"))
    length(wnames)==0 || _checknames(wnames) ||
        throw(ArgumentError("Invalid wnames, must contain either integers or `Symbol`s"))

    # Convert all column indices to Int for merging ynames into wnames
    ynames = (_toint(data, n) for n in ynames)
    wnames = [_toint(data, n) for n in wnames]
    union!(wnames, ynames...)

    ys = AbstractVector[getcolumn(data, n) for n in ynames]
    xs = AbstractVector[getcolumn(data, n) for n in xnames]
    ws = AbstractVector[getcolumn(data, n) for n in wnames]
    if !nocons
        push!(xs, ones(length(ys[1])))
        xnames = (xnames..., :constant)
    end
    B = Vector{Matrix{TF}}(undef, nhorz)
    V = Vector{Matrix{TF}}(undef, nhorz)
    T = Vector{Int}(undef, nhorz)
    # Horizons start from 0
    for h in 0:nhorz-1
        B[h+1], V[h+1], T[h+1] = _lp(ys, xs, ws, nlag, h, vce; TF=TF)
    end

    ynames = [_toname(data, i) for i in ynames]
    xnames = Symbol[_toname(data, i) for i in xnames]
    wnames = [_toname(data, i) for i in wnames]
    return LocalProjectionResult(B, V, T, vce, ynames, xnames, wnames,
        Dict(n=>i for (i,n) in enumerate(ynames)),
        Dict(n=>i for (i,n) in enumerate(vcat(xnames, wnames))),
        nlag, nhorz, nocons)
end

show(io::IO, r::LocalProjectionResult) = print(io, typeof(r).name.name)

function show(io::IO, ::MIME"text/plain", r::LocalProjectionResult)
    print(io, "$(typeof(r).name.name) with $(r.nlag) lag")
    print(io, r.nlag > 1 ? "s " : " ", "over $(r.nhorz) horizon")
    println(io, r.nhorz > 1 ? "s:" : ":")
    print(io, "  outcome name", length(r.ynames) > 1 ? "s:" : ":")
    println(io, (" $n" for n in r.ynames)...)
    if length(r.xnames) > 0
        print(io, "  regressor name", length(r.xnames) > 1 ? "s:" : ":")
        println(io, (" $n" for n in r.xnames)...)
    end
    print(io, "  lagged control name", length(r.wnames) > 1 ? "s:" : ":")
    print(io, (" $n" for n in r.wnames)...)
end
