struct ImpulseResponse{TF<:AbstractFloat, VCE<:CovarianceEstimator} <: StatisticalModel
    B::Vector{TF}
    SE::Vector{TF}
    T::Vector{Int}
    vce::VCE
    yname::Symbol
    xname::Symbol
end

coef(f::ImpulseResponse) = f.B
stderror(f::ImpulseResponse) = f.SE

function confint(f::ImpulseResponse; level::Real=0.9, horz=Colon())
    scale = norminvcdf(1 - (1 - level) / 2)
    se = stderror(f)[horz]
    b = coef(f)[horz]
    return b .- scale .* se, b .+ scale .* se
end

function irf(r::LocalProjectionResult{TF}, yname::Symbol, xwname::Symbol) where TF
    H = length(r.B)
    B = Vector{TF}(undef, H)
    SE = Vector{TF}(undef, H)
    for h in 1:H
        B[h] = coef(r, h, yname, xwname)
        SE[h] = sqrt(vcov(r, h, yname, xwname))
    end
    return ImpulseResponse(B, SE, r.T, r.vce, yname, xwname)
end

function coeftable(f::ImpulseResponse; level::Real=0.90, horz=Colon())
    H = length(f.B)
    if horz !== Colon()
        cf = coef(f)[horz]
        se = stderror(f)[horz]
        cil, ciu = confint(f, level=level, horz=horz)
        cnames = horz.-1
    else
        cf = coef(f)
        se = stderror(f)
        cil, ciu = confint(f, level=level)
        cnames = 0:H-1
    end
    ts = cf ./ se
    pv = 2 .* normccdf.(abs.(ts))
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
