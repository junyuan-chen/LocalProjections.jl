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
    getscore(X::AbstractMatrix, resid::AbstractMatrix)

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
        out[t, j] = X[t,ix] * resid[t,ir]
    end
    return out
end

# Check whether the input data is a column table
function checktable(data)
    Tables.istable(data) ||
        throw(ArgumentError("data of type $(typeof(data)) is not `Tables.jl`-compatible"))
    Tables.columnaccess(data) ||
        throw(ArgumentError("data of type $(typeof(data)) is not a column table"))
end

# Indicate rows with finite and nonmissing data
function _esample!(esample::BitVector, aux::BitVector, v::AbstractVector)
    aux .= isequal.(isfinite.(v), true)
    esample .&= aux
end

"""
    hamilton_filter(y::AbstractVector, h::Integer, p::Integer)
    hamilton_filter(y::AbstractVector, freq::Symbol)

Decompose the time series in `y` into a cyclical component and a trend component
using the method by Hamilton (2018).

Rows in `y` that involve invalid values (e.g., `NaN`, `Inf` and `missing`)
for estimation are skipped.
Returned vectors are of the same length as `y`.

# Arguments
- `h::Integer`: horizon of forcasting; for business cycles, should cover 2 years.
- `p::Integer`: number of lags used for estimation, including the contemporary one.
- `freq::Symbol`: use default values of `h` and `p` suggested by Hamilton (2018) based on data frequency; must be `:m`, `:q` or `:y` for monthly, quarterly or annual data.

# Returns
- `Vector{Float64}`: the cyclical component.
- `Vector{Float64}`: the trend component.
- `BitVector`: indicators for whether an estimate is obtained for each row of input data `y`.

# Reference
Hamilton, James D. 2018. "Why You Should Never Use the Hodrick-Prescott Filter."
The Review of Economics and Statistics 100 (5): 831-843.
"""
function hamilton_filter(y::AbstractVector, h::Integer, p::Integer)
    T = length(y)
    T1 = T - p - h + 1
    h > 0 || throw(ArgumentError("h must be positive; got $h"))
    p > 0 || throw(ArgumentError("p must be positive; got $p"))
    T1 > p + h + 1 || throw(ArgumentError("not enough observations for estimation"))

    # Indicate whether a missing value exists in each row
    esample = trues(T1)
    # A cache for indicators
    aux = BitVector(undef, T1)

    # Construct lag matrix
    X = Matrix{Float64}(undef, T1, p+1)
    X[:,1] .= 1.0
    for j = 1:p
        v = view(y, p-j+1:T-h-j+1)
        _esample!(esample, aux, v)
        copyto!(view(X,esample,j+1), view(v,esample))
    end
    v = view(y, p+h:T)
    _esample!(esample, aux, v)
    Y = v[esample]
    T2 = sum(esample)
    if T2 < T1
        T2 > size(X, 2) || throw(ArgumentError("not enough valid observations left in sample"))
        X = X[esample, :]
    end
    # OLS
    b = X\Y
    Xb = X*b
    trend = fill(NaN, T)
    trend[view(p+h:T, esample)] .= Xb
    cycle = fill(NaN, T)
    cycle[view(p+h:T, esample)] .= Y .- Xb
    # Returned esample has the same length as input data
    prepend!(esample, (false for i in 1:p+h-1))
    return cycle, trend, esample
end

function hamilton_filter(y::AbstractVector, freq::Symbol)
    if freq == :m
        N = 12
    elseif freq == :q
        N = 4
    elseif freq == :y
        N = 1
    else
        throw(ArgumentError("invalid input for freq; must be :m, :q or :y"))
    end
    return hamilton_filter(y, 2*N, N)
end
