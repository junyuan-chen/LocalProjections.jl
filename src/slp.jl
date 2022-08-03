"""
    Ridge{TF<:AbstractFloat} <: RegressionModel

Data from a ridge regression.
"""
struct Ridge{TF<:AbstractFloat} <: RegressionModel
    y::Vector{TF}
    C::Matrix{TF}
    P::Matrix{TF}
    Cy::Vector{TF}
    invCCP::Matrix{TF}
    crossy::TF
    crossC::Matrix{TF}
    Xs::Union{Vector{Matrix{TF}},Nothing}
    λ::TF
    θ::Vector{TF}
    resid::Vector{TF}
    score::Matrix{TF}
    dof_res::TF
    dof_adj::Int
end

modelmatrix(m::Ridge) = m.C
coef(m::Ridge) = m.θ
residuals(m::Ridge) = m.resid
dof_residual(m::Ridge) = m.dof_res - m.dof_adj
dof_tstat(m::Ridge) = dof_residual(m)

show(io::IO, ::Ridge) = print(io, "Ridge regression")

"""
    SearchCriterion

Supertype for all criteria used for model selection.
"""
abstract type SearchCriterion end

show(io::IO, c::SearchCriterion) = print(io, typeof(c).name.name)

"""
    LOOCV <: SearchCriterion

Leave-one-out cross validation.
"""
struct LOOCV <: SearchCriterion end

show(io::IO, ::MIME"text/plain", ::LOOCV) = print(io, "Leave-one-out cross validation")

"""
    GCV <: SearchCriterion

Generalized cross validation.
"""
struct GCV <: SearchCriterion end

show(io::IO, ::MIME"text/plain", ::GCV) = print(io, "Generalized cross validation")

"""
    AIC <: SearchCriterion

Akaike information criterion.
"""
struct AIC <: SearchCriterion end

show(io::IO, ::MIME"text/plain", ::AIC) = print(io, "Akaike information criterion")

"""
    SmoothAlgorithm

Supertype for all algorithms used to compute estimates
needed for selecting the smoothing parameter.
"""
abstract type SmoothAlgorithm end

show(io::IO, a::SmoothAlgorithm) = print(io, typeof(a).name.name)

"""
    DemmlerReinsch <: SmoothAlgorithm

Demmler-Reinsch orthogonalization for fast ridge regressions
with alternative smoothing parameters.
The implementation follows Ruppert et al. (2003).

Once the orthogonalization is solved,
estimates with different smoothing parameters can be generated at very low cost.
The resulting estimates are accurate enough for model selection
and are typically very close to the estimates
generated by the slower [`DirectSolve`](@ref) approach
that directly solves the penalized least squares problem for each parameter.

# Reference
Ruppert, David, M. P. Wand, M., and R. J. Carroll. 2003.
Semiparametric Regression (Cambridge Series in Statistical and Probabilistic Mathematics).
Cambridge: Cambridge University Press.
"""
struct DemmlerReinsch <: SmoothAlgorithm end

show(io::IO, ::MIME"text/plain", ::DemmlerReinsch) =
    print(io, "Demmler-Reinsch orthogonalization")

"""
    DirectSolve <: SmoothAlgorithm

Solve the ridge regressions with alternative smoothing parameters
by directly repeating each penalized least squares problem.

This algorithm is intended to generate the most accurate estimates
for each smoothing parameter
when the computational cost is not a concern.
For a faster alternative, see [`DemmlerReinsch`](@ref).
"""
struct DirectSolve <: SmoothAlgorithm end

"""
    ModelSelection

Supertype for all methods used for generating alternative models.
"""
abstract type ModelSelection end

@fieldequal ModelSelection

show(io::IO, s::ModelSelection) = print(io, typeof(s).name.name)

"""
    GridSearch{A<:SmoothAlgorithm, TF<:AbstractFloat} <: ModelSelection

Exhaustively assess all parameters on a specified grid. See also [`grid`](@ref).
"""
struct GridSearch{A<:SmoothAlgorithm, TF<:AbstractFloat} <: ModelSelection
    v::Vector{TF}
    algo::SmoothAlgorithm
    function GridSearch(v::Vector; algo::SmoothAlgorithm=DemmlerReinsch())
        eltype(v) <: AbstractFloat || (v = convert(Vector{Float64}, v))
        TF = eltype(v)
        thres = eps(TF)
        all(v.>thres) || throw(ArgumentError("only positive values are accepted"))
        return new{typeof(algo),TF}(v, algo)
    end
end

"""
    grid([v]; algo=DemmlerReinsch())

Specify the grid vector `v` used for [`GridSearch`](@ref).
If `v` is not provided, a default grid is used.
The default algorithm used for assessing the alternative parameters
on the grid is [`DemmlerReinsch`](@ref), which is very fast.
"""
grid(v::Vector; kwargs...) = GridSearch(v; kwargs...)
grid(v; kwargs...) = GridSearch([v...]; kwargs...)
grid(; kwargs...) = GridSearch(exp.(LinRange(-6, 12, 50)); kwargs...)

function show(io::IO, ::MIME"text/plain", s::GridSearch{A, TF}) where {A, TF}
    N = length(s.v)
    print(io, typeof(s).name.name, '{', A, ", ", TF, '}')
    println(io, " across ", N, " candidate value", N>1 ? "s:" : ":")
    print(IOContext(io, :compact=>true), "  ", s.v)
end

"""
    ModelSelectionResult

Supertype for all results from model selection.
"""
abstract type ModelSelectionResult end

show(io::IO, sr::ModelSelectionResult) = print(io, typeof(sr).name.name)

"""
    GridSearchResult{TF<:AbstractFloat} <: ModelSelectionResult

Additional results from [`GridSearch`](@ref),
including estimates calculated for each parameter on the grid.

# Fields
- `iopt::Dict{SearchCriterion,Int}`: index of the optimal parameter based on each criterion.
- `θs::Matrix{TF}`: point estimates obtained.
- `loocv::Vector{TF}`: leave-one-out cross validation errors.
- `rss::Vector{TF}`: residual sums of squares.
- `gcv::Vector{TF}`: generalized cross validation errors.
- `aic::Vector{TF}`: Akaike information criterion values.
- `dof_fit::Vector{TF}`: degrees of freedom of the fit.
- `dof_res::Vector{TF}`: residual degrees of freedom.
"""
struct GridSearchResult{TF<:AbstractFloat} <: ModelSelectionResult
    iopt::Dict{SearchCriterion,Int}
    θs::Matrix{TF}
    loocv::Vector{TF}
    rss::Vector{TF}
    gcv::Vector{TF}
    aic::Vector{TF}
    dof_fit::Vector{TF}
    dof_res::Vector{TF}
end

function show(io::IO, ::MIME"text/plain", sr::GridSearchResult)
    N = length(sr.rss)
    print(io, typeof(sr).name.name, " across ", N, " candidate value", N>1 ? "s:" : ":")
    for c in (LOOCV(), GCV(), AIC())
        print(io, "\n  ", rpad(c, 6), "=> ", sr.iopt[c])
    end
end

"""
    SmoothLP{TS<:ModelSelection, TC<:SearchCriterion} <: AbstractEstimator

Smooth local projection method introduced by Barnichon and Brownlees (2019).

The implementation is more computationally efficient than the original Matlab example,
as it does not involve explicit construction of the smoother matrix,
which can be very large.
Regressors that are not B-splines are partialled out before any ridge regression
to save computational cost.
For selecting the smoothing parameter,
the supported criteria are [`LOOCV`](@ref), [`GCV`](@ref) and [`AIC`](@ref).

# Reference
Barnichon, Regis and Christian Brownlees. 2019.
"Impulse Response Estimation by Smooth Local Projections."
The Review of Economics and Statistics 101 (3): 522-530.
"""
struct SmoothLP{TS<:ModelSelection, TC<:SearchCriterion} <: AbstractEstimator
    names::Vector{VarIndex}
    order::Int
    ndiff::Int
    search::TS
    criterion::TC
    fullX::Bool
    function SmoothLP(names::Vector{VarIndex}, order::Int, ndiff::Int, search::TS,
            criterion::TC, fullX::Bool) where {TS<:ModelSelection, TC<:SearchCriterion}
        order > 1 || throw(ArgumentError(
            "the order of the polynomial must be greater than one"))
        ndiff > 0 || throw(ArgumentError(
            "the order of the difference to be penalized must be positive"))
        return new{TS,TC}(names, order, ndiff, search, criterion, fullX)
    end
end

"""
    SmoothLP(names, order::Int=3, ndiff::Int=3; kwargs)

Constructor for [`SmoothLP`](@ref).

# Arguments
- `names`: column indices of variables from the data table to be fit with penalized splines.
- `order`: the highest order of the polynomial term in a spline basis.
- `ndiff`: the order of the finite difference operator used to penalize B-spline coefficients.

# Keywords
- `search::ModelSelection=grid()`: method for generating alternative smoothing parameters.
- `criterion::SearchCriterion=LOOCV()`: criterion for selecting the optimal parameter.
- `fullX::Bool=false`: whether regressor matrices not involved in smoothing are kept.
"""
function SmoothLP(names, order::Int=3, ndiff::Int=3;
        search::ModelSelection=grid(), criterion::SearchCriterion=LOOCV(), fullX::Bool=false)
    names isa VarIndex && (names = (names,))
    names = VarIndex[n for n in names]
    return SmoothLP(names, order, ndiff, search, criterion, fullX)
end

function show(io::IO, ::MIME"text/plain", e::SmoothLP)
    println(io, "Smooth Local Projection:")
    print(io, "  smoothed regressor", length(e.names) > 1 ? "s:" : ":")
    println(io, (" $n" for n in e.names)...)
    println(io, "  polynomial order: ", e.order)
    print(io, "  finite difference order: ", e.ndiff)
end

"""
    SmoothLPResult{TF<:AbstractFloat, TS<:ModelSelectionResult} <: AbstractEstimatorResult

Additional results from estimating a smooth local projection.
See also [`SmoothLP`](@ref) and [`LocalProjectionResult`](@ref).

# Fields
- `θ::Vector{TF}`: coefficient estimates for the B-splines.
- `Σ::Vector{TF}`: variance-covariance estimates for the B-splines.
- `bm::Matrix{TF}`: basis matrix of the B-splines.
- `λ::TF`: the optimal smoothing parameter from model selection.
- `loocv::TF`: leave-one-out cross validation error.
- `rss::TF`: residual sum of squares.
- `gcv::TF`: generalized cross validation error.
- `aic::TF`: Akaike information criterion value.
- `dof_fit::TF`: degree of freedom of the fit.
- `dof_res::TF`: residual degree of freedom.
- `search::TS`: additional results from model selection.
- `m::Ridge{TF}`: data from the ridge regression.
"""
struct SmoothLPResult{TF<:AbstractFloat, TS<:ModelSelectionResult} <: AbstractEstimatorResult
    θ::Vector{TF}
    Σ::Matrix{TF}
    bm::Matrix{TF}
    λ::TF
    loocv::TF
    rss::TF
    gcv::TF
    aic::TF
    dof_fit::TF
    dof_res::TF
    search::TS
    m::Ridge{TF}
end

dof_residual(r::LocalProjectionResult{<:SmoothLP}) = dof_residual(r.estres.m)
dof_tstat(r::LocalProjectionResult{<:SmoothLP}) = dof_residual(r)

function _basismatrix(order::Int, minh::Int, maxh::Int)
    b = BSplineBasis(order+1, minh-order+1:maxh+order-1)
    return basismatrix(b, minh:maxh)[:, order:end-order+1]
end

# Find the regressors to be fit with splines
function _getcols!(est::SmoothLP, data, xnames)
    ix_sm = Vector{Int}(undef, length(est.names))
    for (i, n) in enumerate(est.names)
        fr = findfirst(x->x==_toname(data, n), xnames)
        fr === nothing && throw(ArgumentError(
            "Variable $(n) is not found among x variables"))
        ix_sm[i] = fr
    end
    return ix_sm
end

function _makeYSr(dt, ss, horz; TF=Float64)
    Y, X, CLU, W, T, esample, doffe = _makeYX(dt, horz)
    Tfull = size(dt.ys[1],1)
    ns = length(ss)
    # Filter valid rows within those filtered by _makeYX
    esampleT = trues(T)
    aux = BitVector(undef, T)
    S = Matrix{TF}(undef, T, ns)
    for j in 1:ns
        v = view(vec(ss[j], dt.subset, :x, horz, TF), view(dt.nlag+1:Tfull-horz, esample))
        dt.subset === nothing || j > 1 ||
            (esampleT .&= view(view(dt.subset, dt.nlag+1:Tfull-horz), esample))
        _esample!(esampleT, aux, v)
        copyto!(view(S,esampleT,j), view(v,esampleT))
    end
    T1 = sum(esampleT)
    if T1 < T
        T1 > size(X,2) || throw(ArgumentError(
            "not enough observations for nlag=$(dt.nlag) and horz=$horz"))
        T = T1
        Y = Y[esampleT, :]
        X = X[esampleT, :]
        W isa UnitWeights || (W = W[esampleT, :])
        S = S[esampleT, :]
    end
    W isa UnitWeights || (S .*= sqrt.(W))
    # Partial out X
    YS = [Y S]
    B = X'YS
    ldiv!(cholesky!(X'X), B)
    res = YS - X * B
    return res, X, T, esample, esampleT
end

# Generate the penalty matrix
function _makeP(K::Int, ndiff::Int; TF=Float64)
    D = Matrix{TF}(I, K, K)
    for i in 1:ndiff
        D = diff(D, dims=1)
    end
    return D'D
end

# See appendix B.1 in Ruppert et al. (2003)
function _DemmlerReinsch(C::AbstractMatrix, P::AbstractMatrix, δ::AbstractFloat)
    # Need to approximate C'C with a nonsingular matrix
    CC = cholesky!(C'C+δ*P)
    invR = inv!(CC.U)
    F = svd!(invR'P*invR)
    U = F.U
    s = F.S
    invRU = invR * U
    A = C * invRU
    return A, s, invRU
end

# Compute the diagonal elements in the smoother matrix without forming it
function _Sdiag!(Sdiag::Vector, A::Matrix, d::Vector)
    (T, K) = size(A)
    fill!(Sdiag, 0.0)
    @inbounds for j in 1:K
        dj = d[j]
        for i in 1:T
            Sdiag[i] += (A[i,j]^2)/dj
        end
    end
    return Sdiag
end

function _select(est::SmoothLP{<:GridSearch{DemmlerReinsch}}, y, C, crossy, crossC, Cy, P)
    T, K = size(C)
    A, s, invRU = _DemmlerReinsch(C, P, 1e-10)
    b = A'y
    λs = est.search.v
    N = length(λs)
    TF = eltype(y)
    Sdiag = Vector{TF}(undef, T)
    d = Vector{TF}(undef, length(s))
    bd = similar(d)
    θs = Matrix{TF}(undef, K, N)
    rss = Vector{TF}(undef, N)
    dof_fit = similar(rss)
    cve = similar(y)
    loocv = similar(rss)
    dof_res = similar(rss)
    for i in 1:N
        λ = λs[i]
        d .= 1.0 .+ λ.*s
        bd .= b./d
        rss[i] = crossy - 2.0 * b'bd + bd'bd
        dof_fit[i] = sum(1.0./d)
        _Sdiag!(Sdiag, A, d)
        θ = view(θs,:,i)
        mul!(θ, invRU, bd)
        mul!(cve, C, θ)
        cve .= ((y .- cve)./(1.0 .- Sdiag)).^2
        loocv[i] = sum(cve)
        d .= 1.0./d
        dof_res[i] = T - 2.0*sum(d) + d'd
    end
    gcv = rss./(1.0.-dof_fit./T).^2
    aic = log.(rss) .+ 2.0.*dof_fit./T
    iopt = Dict{SearchCriterion,Int}()
    for (c, v) in zip((LOOCV(), GCV(), AIC()), (loocv, gcv, aic))
        _, il = findmin(v)
        iopt[c] = il
    end
    r = GridSearchResult(iopt, θs, loocv, rss, gcv, aic, dof_fit, dof_res)
    return λs[iopt[est.criterion]], r, Sdiag
end

function _Sdiag!(Sdiag::Vector, C::Matrix, invCCP::Matrix)
    T, K = size(C)
    v = Vector{eltype(C)}(undef, K)
    @inbounds for i in 1:T
        v .= view(C,i,:)
        Sdiag[i] = v' * invCCP * v
    end
    return Sdiag
end

function _select(est::SmoothLP{<:GridSearch{DirectSolve}}, y, C, crossy, crossC, Cy, P)
    T, K = size(C)
    CCP = similar(crossC)
    λs = est.search.v
    N = length(λs)
    TF = eltype(y)
    Sdiag = Vector{TF}(undef, T)
    θs = Matrix{TF}(undef, K, N)
    rss = Vector{TF}(undef, N)
    dof_fit = similar(rss)
    cve = similar(y)
    loocv = similar(rss)
    dof_res = similar(rss)
    for i in 1:N
        λ = λs[i]
        CCP .= crossC .+ λ.*P
        ccp = cholesky!(CCP)
        θ = view(θs,:,i)
        ldiv!(θ, ccp, Cy)
        invCCP = inv!(ccp)
        _Sdiag!(Sdiag, C, invCCP)
        rss[i] = crossy - 2.0 * Cy'θ + θ'*crossC*θ
        dof_fit[i] = sum(Sdiag)
        mul!(cve, C, θ)
        cve .= ((y .- cve)./(1.0 .- Sdiag)).^2
        loocv[i] = sum(cve)
        dof_res[i] = T - 2.0*dof_fit[i] + Sdiag'Sdiag
    end
    gcv = rss./(1.0.-dof_fit./T).^2
    aic = log.(rss) .+ 2.0.*dof_fit./T
    iopt = Dict{SearchCriterion,Int}()
    for (c, v) in zip((LOOCV(), GCV(), AIC()), (loocv, gcv, aic))
        _, il = findmin(v)
        iopt[c] = il
    end
    r = GridSearchResult(iopt, θs, loocv, rss, gcv, aic, dof_fit, dof_res)
    return λs[iopt[est.criterion]], r, Sdiag
end

function _est(est::SmoothLP, data, xnames, ys, xs, ws, sts, fes, clus, pw, nlag,
        minhorz, nhorz, vce, subset, groups, iv, ix_iv, nendo, niv, yfs, xfs,
        firststagebyhorz, testweakiv; TF=Float64)
    length(ys) > 1 && throw(ArgumentError("accept only one outcome variable"))
    vce isa ClusterCovariance && throw(ArgumentError(
        "cluster-robust VCE is not supported for smoothed local projection"))
    ix_sm = _getcols!(est, data, xnames)
    ix_nsm = setdiff(1:length(xs), ix_sm)
    nx = length(ix_nsm) + length(ws)*nlag
    YSr = Vector{Matrix{TF}}(undef, nhorz)
    Xs = Vector{Matrix{TF}}(undef, nhorz)
    T = Vector{Int}(undef, nhorz)
    firststagebyhorz = iv !== nothing && firststagebyhorz
    if !firststagebyhorz && !any(x->x isa Cum, view(xs, ix_nsm))
        xs_s = Any[xs[i] for i in ix_nsm]
        dt = LPData(ys, xs_s, ws, sts, fes, clus, pw, nlag, minhorz, subset, groups, TF)
    end
    F_kps = firststagebyhorz ? Vector{Float64}(undef, nhorz) : nothing
    p_kps = firststagebyhorz ? Vector{Float64}(undef, nhorz) : nothing
    for h in minhorz:minhorz+nhorz-1
        i = h - minhorz + 1
         # Handle cases where all data need to be regenerated for each horizon
        if firststagebyhorz
            fitted, F_kps[i], p_kps[i] = _firststage(nendo, niv, yfs, xfs,
                ws, sts, fes, clus, pw, nlag, h, subset, groups, testweakiv, vce; TF=TF)
            xs[ix_iv] .= fitted
            xs_s = Any[xs[i] for i in ix_nsm]
            dt = LPData(ys, xs_s, ws, sts, fes, clus, pw, nlag, h, subset, groups, TF)
        elseif any(x->x isa Cum, view(xs, ix_nsm))
            xs_s = Any[xs[i] for i in ix_nsm]
            dt = LPData(ys, xs_s, ws, sts, fes, clus, pw, nlag, h, subset, groups, TF)
        end
        # xs could be changed by first-stage regression
        ss = view(xs, ix_sm)
        YSr[i], Xs[i], T[i], _, _ = _makeYSr(dt, ss, h; TF=TF)
    end
    Tall = sum(T)
    nys = size(YSr[1], 2)
    ns = nys - 1
    bm = _basismatrix(est.order, minhorz, minhorz+nhorz-1)
    nb = size(bm, 2)
    y = Vector{TF}(undef, Tall)
    C = Matrix{TF}(undef, Tall, ns*nb)
    ir = 0
    @inbounds for h in 1:nhorz
        copyto!(view(y,ir+1:ir+T[h]), view(YSr[h],:,1))
        for j in 1:ns
            C[ir+1:ir+T[h],1+nb*(j-1):nb*j] .= view(bm,h,:)'.*view(YSr[h],:,1+j)
        end
        ir += T[h]
    end
    crossy = y'y
    crossC = C'C
    Cy = C'y
    P = _makeP(ns*nb, est.ndiff; TF=TF)
    λ, sr, Sdiag = _select(est, y, C, crossy, crossC, Cy, P)
    CCP = cholesky!(crossC + λ*P)
    if est.search isa GridSearch{DirectSolve}
        l = sr.iopt[est.criterion]
        θ, loocv, rss, gcv, aic, dof_fit, dof_res = sr.θs[:,l], sr.loocv[l],
            sr.rss[l], sr.gcv[l], sr.aic[l], sr.dof_fit[l], sr.dof_res[l]
    else
        # Recalculate to ensure the accuracy of final estimates
        θ = CCP \ (Cy)
        rss = crossy - 2.0 * Cy'θ + θ'*crossC*θ
    end
    resid = C * θ
    resid .= y .- resid
    invCCP = inv!(CCP)
    if !(est.search.algo isa DirectSolve)
        _Sdiag!(Sdiag, C, invCCP)
        dof_fit = sum(Sdiag)
        dof_res = Tall - 2.0*dof_fit + Sdiag'Sdiag
        loocv = sum((resid./(1.0.-Sdiag)).^2)
        gcv = rss/(1.0-dof_fit/Tall)^2
        aic = log(rss) + 2.0*dof_fit/Tall
    end
    B = permutedims(reshape(bm*reshape(θ,nb,ns), nhorz, ns, 1), (2,3,1))
    m = Ridge(y, C, P, Cy, invCCP, crossy, crossC, est.fullX ? Xs : nothing, λ, θ, resid,
        getscore(C, resid), dof_res, nx*nhorz)
    bms = repeat(bm, 1, ns)
    Σ = vcov(m, vce)
    Vfull = bms * Σ * bms'
    V = Array{TF,3}(undef, ns, ns, nhorz)
    inds = [nhorz*(n-1) for n in 1:ns]
    for h in 1:nhorz
        inds .+= 1
        V[:,:,h] = reshape(view(Vfull, inds, inds), ns, ns, 1)
    end
    slpr = SmoothLPResult(θ, Σ, bm, λ, loocv, rss, gcv, aic, dof_fit, dof_res, sr, m)
    return B, V, T, slpr, F_kps, p_kps
end

function lp(r::LocalProjectionResult{<:SmoothLP}, vce::CovarianceEstimator)
    ns = size(r.V, 1)
    s = r.estres
    Σ = vcov(s.m, vce)
    bms = repeat(s.bm, 1, ns)
    Vfull = bms * Σ * bms'
    V = similar(r.V)
    nhorz = length(r.T)
    inds = [nhorz*(n-1) for n in 1:ns]
    for h in 1:nhorz
        inds .+= 1
        V[:,:,h] = reshape(view(Vfull, inds, inds), ns, ns, 1)
    end
    slpr = SmoothLPResult(s.θ, Σ, s.bm, s.λ, s.loocv, s.rss, s.gcv, s.aic,
        s.dof_fit, s.dof_res, s.search, s.m)
    return LocalProjectionResult(r.B, V, r.T, r.est, slpr, vce,
        r.ynames, r.xnames, r.wnames, r.stnames, r.fenames, r.lookupy, r.lookupx, r.lookupw,
        r.panelid, r.panelweight, r.nlag, r.minhorz, r.subset,
        r.normnames, r.normtars, r.normmults,
        r.endonames, r.ivnames, r.firststagebyhorz, r.F_kp, r.p_kp, r.nocons)
end

"""
    lp(r::LocalProjectionResult{<:SmoothLP}, λ::Real; vce=nothing)

Reestimate the smooth local projection with the smoothing parameter `λ`.
Optionally, an alternative variance-covariance estimator may be specified
with the keyword `vce`.
"""
function lp(r::LocalProjectionResult{<:SmoothLP}, λ::Real;
        vce::Union{CovarianceEstimator, Nothing}=nothing)
    m = r.estres.m
    λ = convert(typeof(m.λ), λ)
    Tall = length(m.y)
    CCP = cholesky!(m.crossC + λ*m.P)
    θ = CCP \ (m.Cy)
    rss = m.crossy - 2.0 * m.Cy'θ + θ'*m.crossC*θ
    resid = m.C * θ
    resid .= m.y .- resid
    invCCP = inv!(CCP)
    Sdiag = similar(m.y)
    _Sdiag!(Sdiag, m.C, invCCP)
    dof_fit = sum(Sdiag)
    dof_res = Tall - 2.0*dof_fit + Sdiag'Sdiag
    loocv = sum((resid./(1.0.-Sdiag)).^2)
    gcv = rss/(1.0-dof_fit/Tall)^2
    aic = log(rss) + 2.0*dof_fit/Tall

    bm = r.estres.bm
    nb = size(bm, 2)
    ns = length(r.est.names)
    nhorz = length(r.T)
    B = permutedims(reshape(bm*reshape(θ,nb,ns), nhorz, ns, 1), (2,3,1))
    m1 = Ridge(m.y, m.C, m.P, m.Cy, invCCP, m.crossy, m.crossC, m.Xs, λ, θ, resid,
        getscore(m.C, resid), dof_res, m.dof_adj)
    bms = repeat(bm, 1, ns)
    vce1 = vce===nothing ? r.vce : vce
    Σ = vcov(m1, vce1)
    Vfull = bms * Σ * bms'
    V = similar(r.V)
    inds = [nhorz*(n-1) for n in 1:ns]
    for h in 1:nhorz
        inds .+= 1
        V[:,:,h] = reshape(view(Vfull, inds, inds), ns, ns, 1)
    end
    slpr = SmoothLPResult(θ, Σ, bm, λ, loocv, rss, gcv, aic, dof_fit, dof_res,
        r.estres.search, m1)
    return LocalProjectionResult(B, V, r.T, r.est, slpr, vce1,
        r.ynames, r.xnames, r.wnames, r.stnames, r.fenames, r.lookupy, r.lookupx, r.lookupw,
        r.panelid, r.panelweight, r.nlag, r.minhorz, r.subset,
        r.normnames, r.normtars, r.normmults,
        r.endonames, r.ivnames, r.firststagebyhorz, r.F_kp, r.p_kp, r.nocons)
end

_estimatortitle(::LocalProjectionResult{<:SmoothLP}) = "Smooth Local Projection"

function _estimatorinfo(r::LocalProjectionResult{<:SmoothLP}, halfwidth::Int)
    info = ["Smoothing parameter" => TestStat(r.estres.λ),
        "Smoothed regressor" => join(r.est.names, " "),
        "Polynomial order" => r.est.order,
        "Finite difference order" => r.est.ndiff,
        "Selection criterion" => r.est.criterion,
        "Selection algorithm" => r.est.search.algo,
        "Leave-one-out CV" => TestStat(r.estres.loocv),
        "Generalized CV" => TestStat(r.estres.gcv),
        "Akaike information" => TestStat(r.estres.aic),
        "Residual sum of squares" => TestStat(r.estres.rss)]
    return info
end
