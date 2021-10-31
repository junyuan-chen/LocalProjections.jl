"""
    ImpulseResponse{TF<:AbstractFloat, VCE<:CovarianceEstimator} <: StatisticalModel

Estimates for an impulse response function.
"""
struct ImpulseResponse{TF<:AbstractFloat, VCE<:CovarianceEstimator} <: StatisticalModel
    B::Vector{TF}
    SE::Vector{TF}
    T::Vector{Int}
    vce::VCE
    yname::VarName
    xname::VarName
    minhorz::Int
end

coef(f::ImpulseResponse) = f.B
stderror(f::ImpulseResponse) = f.SE

function confint(f::ImpulseResponse; level::Real=0.9, horz=Colon())
    scale = criticalvalue.(Ref(f.vce), level, view(f.T, horz))
    se = stderror(f)[horz]
    b = coef(f)[horz]
    return b .- scale .* se, b .+ scale .* se
end

"""
    irf(r::LocalProjectionResult, yname::VarName, xwname::VarName; lag::Int=0)

Return the [`ImpulseResponse`](@ref) of outcome variable `yname`
with respect to variable `xwname` based on the estimation result `r`.
If `lag` is not specified, `xwname` is assumed to be contemporaneous.
"""
function irf(r::LocalProjectionResult, yname::VarName, xwname::VarName; lag::Int=0)
    TF = typeof(r).parameters[4]
    H = length(r.T)
    B = Vector{TF}(undef, H)
    SE = Vector{TF}(undef, H)
    for h in 1:H
        B[h] = coef(r, h, xwname, yname=yname, lag=lag)
        SE[h] = sqrt(vcov(r, h, xwname, yname1=yname, lag1=lag))
    end
    return ImpulseResponse(B, SE, r.T, r.vce, yname, xwname, r.minhorz)
end

function coeftable(f::ImpulseResponse;
        level::Real=0.90, horz::Union{Integer,AbstractVector,Colon}=Colon())
    H = length(f.B)
    if horz !== Colon()
        cf = coef(f)[horz]
        se = stderror(f)[horz]
        cil, ciu = confint(f, level=level, horz=horz)
        cnames = horz .+ f.minhorz .-1
    else
        cf = coef(f)
        se = stderror(f)
        cil, ciu = confint(f, level=level)
        cnames = f.minhorz:f.minhorz+H-1
    end
    ts = cf ./ se
    pv = pvalue.(Ref(f.vce), ts, view(f.T, horz))
    levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
    return CoefTable(Vector[cf, se, ts, pv, cil, ciu],
        ["Estimate", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        ["$(cnames[i])" for i = 1:length(cf)], 4, 3)
end

show(io::IO, f::ImpulseResponse) = print(io, typeof(f).name.name)

function show(io::IO, ::MIME"text/plain", f::ImpulseResponse)
    H = length(f.B)
    nr, nc = displaysize(io)
    print(io, typeof(f).name.name, " with $H horizon")
    println(io, H > 1 ? "s:" : ":")
    if H > nr-8
        show(io, coeftable(f, horz=1:nr-9))
        printstyled(io, "\n", H-nr+9, " rows of estimates omitted"; color=:cyan)
    else
        show(io, coeftable(f))
    end
end
