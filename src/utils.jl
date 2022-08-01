"""
    @fieldequal Supertype

Define a method of `==` for all subtypes of `Supertype`
such that `==` returns true if each pair of the field values
from two instances are equal by `==`.
"""
macro fieldequal(Supertype)
    return esc(quote
        function ==(x::T, y::T) where T <: $Supertype
            f = fieldnames(T)
            getfield.(Ref(x),f) == getfield.(Ref(y),f)
        end
    end)
end

"""
    kron_fastl(A::AbstractMatrix, B::AbstractMatrix)

Compute `kron(I(n), A) * B` without explicitely calculating the Kronecker product
where `I(n)` is an identity matrix with
`n` being the ratio between `size(B, 1)` and `size(A, 2)`.
See also [`kron_fastr`](@ref).

# Reference
Acklam, Peter J. 2003. "MATLAB Array Manipulation Tips and Tricks."
"""
function kron_fastl(A::AbstractMatrix, B::AbstractMatrix)
    p, q = size(A)
    qn, m = size(B)
    n = qn ÷ q
    return reshape(A*reshape(B, q, n*m), p*n, m)
end

"""
    kron_fastr(A::AbstractMatrix, B::AbstractMatrix)

Compute `kron(A, I(n)) * B` without explicitely calculating the Kronecker product
where `I(n)` is an identity matrix with
`n` being the ratio between `size(B, 1)` and `size(A, 2)`.
See also [`kron_fastl`](@ref).

# Reference
Acklam, Peter J. 2003. "MATLAB Array Manipulation Tips and Tricks."
"""
function kron_fastr(A::AbstractMatrix, B::AbstractMatrix)
    p, q = size(A)
    qn, m = size(B)
    n = qn ÷ q
    return reshape(reshape(B', n*m, q)*A', m, p*n)'
end

"""
    getscore(X::AbstractMatrix, resid::AbstractVecOrMat)

Compute the regression scores (the product of the residuals and regressors)
with possibly multiple outcome variables.
"""
function getscore(X::AbstractMatrix, resid::AbstractMatrix)
    T, K = size(X)
    N = size(resid, 2)
    out = similar(X, T, K*N)
    @inbounds for (t, j) in Base.product(1:T, 1:K*N)
        ix = (j-1) ÷ N + 1
        ir = j % N
        ir == 0 && (ir = N)
        out[t,j] = X[t,ix] * resid[t,ir]
    end
    return out
end

getscore(X::AbstractMatrix, resid::AbstractVector) = X .* resid

# Check whether the input data is a column table
function checktable(data)
    Tables.istable(data) ||
        throw(ArgumentError("data of type $(typeof(data)) is not `Tables.jl`-compatible"))
    Tables.columnaccess(data) ||
        throw(ArgumentError("data of type $(typeof(data)) is not a column table"))
end

# Indicate rows with finite and nonmissing data
function _esample!(esample::AbstractVector{Bool}, aux::AbstractVector{Bool},
        v::AbstractVector)
    aux .= isequal.(isfinite.(v), true)
    esample .&= aux
end

"""
    hamilton_filter(y::AbstractVector, h::Integer, p::Integer; subset)
    hamilton_filter(y::AbstractVector, freq::Symbol; subset)

Decompose the time series in `y` into a cyclical component and a trend component
using the method by Hamilton (2018).

Rows in `y` that involve invalid values (e.g., `NaN`, `Inf` and `missing`)
for estimation are skipped.
Returned vectors are of the same length as `y`.

# Arguments
- `y::AbstractVector`: vector storing the time series to be filtered.
- `h::Integer`: horizon of forcasting; for business cycles, should cover 2 years.
- `p::Integer`: number of lags used for estimation, including the contemporary one.
- `freq::Symbol`: use default values of `h` and `p` suggested by Hamilton (2018) based on data frequency; must be `:m`, `:q` or `:y` for monthly, quarterly or annual data.

# Keywords
- `subset::Union{BitVector,Nothing}=nothing`: Boolean indices of rows in `y` to be filtered.

# Returns
- `Vector{Float64}`: the cyclical component.
- `Vector{Float64}`: the trend component.
- `BitVector`: indicators for whether an estimate is obtained for each row of input data `y`.

# Reference
Hamilton, James D. 2018. "Why You Should Never Use the Hodrick-Prescott Filter."
The Review of Economics and Statistics 100 (5): 831-843.
"""
function hamilton_filter(y::AbstractVector, h::Integer, p::Integer;
        subset::Union{BitVector,Nothing}=nothing)
    Tfull = length(y)
    T = Tfull - p - h + 1
    h > 0 || throw(ArgumentError("h must be positive; got $h"))
    p > 0 || throw(ArgumentError("p must be positive; got $p"))
    T > p + h + 1 || throw(ArgumentError("not enough observations for estimation"))

    # Indicate whether a missing value exists in each row
    esample = trues(T)
    # A cache for indicators
    aux = BitVector(undef, T)

    # Construct lag matrix
    X = Matrix{Float64}(undef, T, p+1)
    X[:,1] .= 1.0
    for j = 1:p
        v = view(y, p-j+1:Tfull-h-j+1)
        subset === nothing || (esample .&= view(subset, p-j+1:Tfull-h-j+1))
        _esample!(esample, aux, v)
        copyto!(view(X,esample,j+1), view(v,esample))
    end
    v = view(y, p+h:Tfull)
    subset === nothing || (esample .&= view(subset, p+h:Tfull))
    _esample!(esample, aux, v)
    Y = v[esample]
    T1 = sum(esample)
    if T1 < T
        T1 > size(X, 2) || throw(ArgumentError("not enough valid observations left in sample"))
        X = X[esample, :]
    end
    # OLS
    b = X \ Y
    Xb = X * b
    trend = fill(NaN, Tfull)
    trend[view(p+h:Tfull, esample)] .= Xb
    cycle = fill(NaN, Tfull)
    cycle[view(p+h:Tfull, esample)] .= Y .- Xb
    # Returned esample has the same length as input data
    prepend!(esample, falses(p+h-1))
    return cycle, trend, esample
end

function hamilton_filter(y::AbstractVector, freq::Symbol;
        subset::Union{BitVector,Nothing}=nothing)
    if freq == :m
        N = 12
    elseif freq == :q
        N = 4
    elseif freq == :y
        N = 1
    else
        throw(ArgumentError("invalid input for freq; must be :m, :q or :y"))
    end
    return hamilton_filter(y, 2*N, N, subset=subset)
end

"""
    TransformedVar{T}

Supertype for all types used to specify variable transformation.
"""
abstract type TransformedVar{T} end

_geto(v::TransformedVar) = getfield(v, :o)
size(v::TransformedVar, dim) = size(_geto(v), dim)
length(v::TransformedVar) = length(_geto(v))
vec(v::Union{AbstractVector,TransformedVar}, subset::Union{AbstractVector{Bool},Nothing},
    vartype::Symbol, horz::Int, TF::Type) = v

"""
    Cum{T,S} <: TransformedVar{T}

Cumulatively summed variable for estimating cumulative impulse response.
State dependency is allowed by specifying a second variable.
"""
struct Cum{T,S} <: TransformedVar{T}
    o::T
    s::S
end

Cum(n::Symbol) = Cum(n, nothing)
Cum(i::Integer) = Cum(Int(i), nothing)
Cum(v::AbstractVector) = Cum(v, nothing)

getcolumn(data, c::Cum) =
    Cum(getcolumn(data, _geto(c)), c.s===nothing ? nothing : getcolumn(data, c.s))

view(c::Cum{<:AbstractVector}, ids) = Cum(view(_geto(c), ids), c.s)

function vec(c::Cum{<:AbstractVector}, subset::Union{<:AbstractVector{Bool},Nothing},
        vartype::Symbol, horz::Int, TF::Type)
    v = c.o
    out = zeros(TF, length(v))
    T = length(v) - horz
    na = convert(TF, NaN)
    ts = vartype === :y ? horz : 0
    if vartype === :y || c.s === nothing
        if subset === nothing
            for h in 0:horz
                for t in 1:T
                    out[ts+t] += coalesce(v[t+h], na)
                end
            end
        else
            for h in 0:horz
                for t in 1:T
                    out[ts+t] += ifelse(subset[t+h], coalesce(v[t+h], na), na)
                end
            end
        end
    else
        if subset === nothing
            for h in 0:horz
                for t in 1:T
                    out[t] += coalesce(c.s[t]*v[t+h], na)
                end
            end
        else
            for h in 0:horz
                for t in 1:T
                    out[t] += ifelse(subset[t+h], coalesce(c.s[t]*v[t+h], na), na)
                end
            end
        end
    end
    if vartype == :y
        out[1:horz] .= na
    else
        out[T+1:end] .= na
    end
    return out
end

_toint(data, ::Nothing) = nothing
_toint(data, c::Cum) = Cum(_toint(data, _geto(c)), _toint(data, c.s))
_toname(data, ::Nothing) = nothing
_toname(data, c::Cum) = Cum(_toname(data, _geto(c)), _toname(data, c.s))

show(io::IO, c::Cum{<:Union{Int,Symbol}}) =
    print(io, typeof(c).name.name, "(", _geto(c), c.s===nothing ? ")" : ", $(c.s))")

# Get indices for consecutive rows with the same values
function _group(col::AbstractVector)
    inds = Vector{Int}[]
    vlast = nothing
    @inbounds for (i, v) in enumerate(col)
        if v == vlast
            push!(last(inds), i)
        else
            push!(inds, Int[i])
            vlast = v
        end
    end
    return inds
end

# Residualize columns in Y and X with weights W for fixed effects FE
function _feresiduals!(Y::AbstractVecOrMat, X::AbstractMatrix, FE::Vector{FixedEffect},
        W::AbstractWeights; nfethreads::Int=Threads.nthreads(),
        fetol::Real=1e-8, femaxiter::Int=10000)
    feM = AbstractFixedEffectSolver{Float64}(FE, W, Val{:cpu}, nfethreads)
    M = Combination(Y, X)
    _, iters, convs = solve_residuals!(M, feM;
        tol=fetol, maxiter=femaxiter, progress_bar=false)
    iter = maximum(iters)
    conv = all(convs)
    conv || @warn "no convergence of fixed effect solver in $(iter) iterations"
end

"""
    datafile(name::Union{Symbol,String})

Return the file path of the example data file named `name`.csv.gz.
"""
datafile(name::Union{Symbol,String}) = (@__DIR__)*"/../data/$(name).csv.gz"
