struct SimpleVCE <: CovarianceEstimator end

function vcov(m::OLS, ::SimpleVCE)
    T, K = size(m.x)
    return kron(m.invxx, m.resid'm.resid)./(T-K)
end

struct HRVCE <: CovarianceEstimator end

function vcov(m::OLS, ::HRVCE)
    T, K = size(m.x)
    return T/(T-K).*kron_fastr(m.invxx, kron_fastr(m.invxx, m.score'm.score)')'
end

