vcov(m, ::Nothing) = nothing

"""
    criticalvalue(vce::CovarianceEstimator, level::Real, T::Int, dofr::Int)

Return the critical value for `vce` at the given `level` when the sample has `T` periods
and the residual degrees of freedom is `dofr`.
"""
criticalvalue(::CovarianceEstimator, level::Real, T::Int, dofr::Int) =
    tdistinvccdf(dofr, (1 - level) / 2)

"""
    pvalue(vce::CovarianceEstimator, ts::Real, T::Int, dofr::Int)

Return the p-value for test statistic `ts` associated with `vce`
when the sample has `T` periods and the residual degrees of freedom is `dofr`.
"""
pvalue(::CovarianceEstimator, ts::Real, T::Int, dofr::Int) = 2 * tdistccdf(dofr, abs(ts))

"""
    SimpleVCE <: CovarianceEstimator

Simple variance-covariance estimator.
"""
struct SimpleVCE <: CovarianceEstimator end

function vcov(m::OLS, ::SimpleVCE)
    T, K = size(m.X)
    return kron(m.invXX, m.resid'm.resid)./(T-K)
end

show(io::IO, ::MIME"text/plain", ::SimpleVCE) =
    print(io, "Simple variance-covariance estimator")

"""
    HRVCE <: CovarianceEstimator

Eicker-Huber-White heteroskedasticity-robust variance-covariance estimator.
"""
struct HRVCE <: CovarianceEstimator end

function vcov(m::OLS, ::HRVCE)
    T, K = size(m.X)
    return T/(T-K).*kron_fastr(m.invXX, kron_fastr(m.invXX, m.score'm.score)')'
end

function vcov(m::Ridge, ::HRVCE)
    T = size(m.C, 1)
    S = m.score'm.score
    return T/(m.dof_res-m.dof_adj) * m.invCCP * S * m.invCCP
end

show(io::IO, ::MIME"text/plain", ::HRVCE) =
    print(io, "Heteroskedasticity-robust variance-covariance estimator")

"""
    LongRunVariance

Supertype for all long-run variance estimators.
"""
abstract type LongRunVariance end

"""
    HARVCE{LR<:LongRunVariance} <: CovarianceEstimator

Heteroskedasticity-autocorrelation-robust variance-covariance estimator
with a long-run variance estimator of type `LR`.

# Fields
- `lr::LR`: the long-run variance estimator.
- `bw::Function`: bandwidth as a function of the number of periods in sample.
- `cv::Function`: critical value as a function of the level of the test and the bandwidth.
- `pv::Function`: p-value as a function of the test statistic and the bandwidth.
"""
struct HARVCE{LR<:LongRunVariance} <: CovarianceEstimator
    lr::LR
    bw::Function
    cv::Function
    pv::Function
end

"""
    EqualWeightedCosine <: LongRunVariance

Equal-weighted cosine transform. The alias [`EWC`](@ref) is provided for convenience.

# Reference
Lazarus, Eben, Daniel J. Lewis, James H. Stock, and Mark W. Watson. 2018.
"HAR Inference: Recommendations for Practice."
Journal of Business & Economic Statistics 36 (4): 541-559.
"""
struct EqualWeightedCosine <: LongRunVariance end

"""
    EWC <: LongRunVariance

An alias for [`EqualWeightedCosine`](@ref).
"""
const EWC = EqualWeightedCosine

HARVCE(lr::EqualWeightedCosine; bw=_ewcbw, cv=_ewccv, pv=_ewcpv) = HARVCE(lr, bw, cv, pv)

_ewcbw(T::Integer) = round(Int, 0.4*T^(2/3))
_ewccv(level::Real, bw::Int) = tdistinvccdf(bw, (1.0-level)/2.0)
_ewcpv(ts::Real, bw::Int) = 2.0*tdistccdf(bw, abs(ts))

function lrv(vce::HARVCE{EqualWeightedCosine}, sc::AbstractMatrix)
    T, K = size(sc)
    TF = eltype(sc)
    bw = vce.bw(T)
    Z = zeros(Complex{TF}, 4*T, K)
    inds = iseven.(1:2*T)
    copyto!(view(view(Z, 1:2*T, :), inds, :), sc)
    fft!(Z, 1)
    rffts = real.(view(Z, 2:bw+1, :))
    return 2.0/bw * (rffts'rffts)
end

show(io::IO, ::MIME"text/plain", ::EqualWeightedCosine) =
    print(io, "Equal-weighted cosine transform")

function vcov(m::OLS, vce::HARVCE)
    T, K = size(m.X)
    return T/(T-K).*kron_fastr(m.invXX, kron_fastr(m.invXX, lrv(vce, m.score))')'
end

function vcov(m::Ridge, vce::HARVCE)
    T = size(m.C, 1)
    return T/(m.dof_res-m.dof_adj) * m.invCCP * lrv(vce, m.score) * m.invCCP
end

criticalvalue(vce::HARVCE, level::Real, T::Int, dofr::Int) = vce.cv(level, vce.bw(T))
pvalue(vce::HARVCE, ts::Real, T::Int, dofr::Int) = vce.pv(ts, vce.bw(T))

function show(io::IO, ::MIME"text/plain", vce::HARVCE)
    println(io, "Heteroskedasticity-autocorrelation-robust variance-covariance estimator:")
    print(io, "  Long-run variance estimator: ", sprint(show, MIME("text/plain"), vce.lr))
end

# Allow OLS to work with ClusterCovariance from Vcov.jl
function vcov(m::OLS, vce::ClusterCovariance)
    pinvertible(Symmetric(m.invXX * S_hat(m, vce) * m.invXX))
end
