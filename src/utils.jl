"""
    kron_fastl(A::AbstractMatrix, B::AbstractMatrix)

Compute `kron(I(n), A) * B` without explicitely calculating the Kronecker product
where `I(n)` is an identity matrix with
`n` being the ratio between `size(B, 1)` and `size(A, 2)`.
See also [`kron_fastr`](@ref).

# Reference
Peter J. Acklam. 2003. "MATLAB Array Manipulation Tips and Tricks."
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
Peter J. Acklam. 2003. "MATLAB Array Manipulation Tips and Tricks."
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
