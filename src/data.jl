struct PartialOLS{TF<:AbstractFloat}
    X::Matrix{TF}
    crossX::Matrix{TF}
    coef::Matrix{TF}
    resid::Matrix{TF}
end

function PartialOLS(T, NY, NX, NG, TF=Float64)
    X = Matrix{TF}(undef, T, NX)
    crossX = Matrix{TF}(undef, NX, NX)
    coef = Matrix{TF}(undef, NX, NG*NY)
    resid = Matrix{TF}(undef, T, NY)
    return PartialOLS(X, crossX, coef, resid)
end

"""
    LPData{TF<:AbstractFloat,TG<:Union{Vector,Nothing},TW<:Union{AbstractVector,Nothing}}

Raw data collected for generating data matrices for local projection estimation.
"""
struct LPData{TF<:AbstractFloat, TG<:Union{Vector,Nothing},
        TW<:Union{AbstractVector,Nothing}, TE<:Union{BitVector,Nothing}}
    ys::Vector{Any}
    wgs::Vector{Any}
    fes::Vector{Any}
    ipanelfe::Union{Int,Nothing}
    clus::Vector{Any}
    pw::TW
    nlag::Int
    minhorz::Int
    subset::Union{BitVector,Nothing}
    Xfull::Matrix{TF}
    esampleTfull::TE
    groups::TG
end

function _fillX!(X, esampleT::AbstractVector{Bool}, aux::AbstractVector{Bool},
        xs, ws, sts, fes, clus, pw, nlag, horz, subset, TF)
    Tfull = size(ws[1],1)
    nx = length(xs)
    nw = length(ws)
    nfe = length(fes)
    nclu = length(clus)
    for j in 1:nx
        v = view(vec(xs[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
        subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull-horz))
        _esample!(esampleT, aux, v)
        copyto!(view(X,esampleT,j), view(v,esampleT))
    end
    if sts === nothing
        for j in 1:nw
            for l in 1:nlag
                v = view(vec(ws[j], subset, :x, horz, TF), nlag+1-l:Tfull-horz-l)
                subset === nothing || j > 1 ||
                    (esampleT .&= view(subset, nlag+1-l:Tfull-horz-l))
                _esample!(esampleT, aux, v)
                # Variables with the same lag are put together
                copyto!(view(X,esampleT,nx+(l-1)*nw+j), view(v,esampleT))
            end
        end
    else
        ns = length(sts)
        for s in 1:ns
            sst = view(vec(sts[s], subset, :x, horz, TF), nlag+1:Tfull-horz)
            _esample!(esampleT, aux, sst)
            for j in 1:nw
                wj = vec(ws[j], subset, :x, horz, TF)
                for l in 1:nlag
                    v = view(wj, nlag+1-l:Tfull-horz-l)
                    subset === nothing || j > 1 || s > 1 ||
                        (esampleT .&= view(subset, nlag+1-l:Tfull-horz-l))
                    s > 1 || _esample!(esampleT, aux, v)
                    X[esampleT, nx+(l-1)*nw*ns+(j-1)*ns+s] .=
                        view(sst,esampleT) .* view(v,esampleT)
                end
            end
        end
    end
    for j in 1:nfe
        v = view(vec(fes[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
        subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull-horz))
        _esample!(esampleT, aux, v)
    end
    for j in 1:nclu
        v = view(vec(clus[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
        subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull-horz))
        _esample!(esampleT, aux, v)
    end
    if pw !== nothing
        v = view(vec(pw, subset, :x, horz, TF), nlag+1:Tfull-horz)
        subset === nothing || (esampleT .&= view(subset, nlag+1:Tfull-horz))
        _esample!(esampleT, aux, v)
    end
end

# A version that assumes all data rows are valid
function _fillX!(X, xs, ws, sts, nlag, horz, TF)
    Tfull = size(ws[1],1)
    nx = length(xs)
    nw = length(ws)
    for j in 1:nx
        v = view(vec(xs[j], nothing, :x, horz, TF), nlag+1:Tfull-horz)
        copyto!(view(X,:,j), v)
    end
    if sts === nothing
        for j in 1:nw
            for l in 1:nlag
                v = view(vec(ws[j], nothing, :x, horz, TF), nlag+1-l:Tfull-horz-l)
                # Variables with the same lag are put together
                copyto!(view(X,:,nx+(l-1)*nw+j), v)
            end
        end
    else
        ns = length(sts)
        for s in 1:ns
            sst = view(vec(sts[s], nothing, :x, horz, TF), nlag+1:Tfull-horz)
            for j in 1:nw
                wj = vec(ws[j], nothing, :x, horz, TF)
                for l in 1:nlag
                    v = view(wj, nlag+1-l:Tfull-horz-l)
                    X[:, nx+(l-1)*nw*ns+(j-1)*ns+s] .= sst .* v
                end
            end
        end
    end
end

function _checklpdata(ys, ws, nlag)
    length(ys) > 0 || throw(ArgumentError("ys cannot be empty"))
    nlag > 0 || throw(ArgumentError("nlag must be at least 1"))
    length(ws) > 0 || throw(ArgumentError("ws cannot be empty"))
end

function LPData(ys, xs, ws, wgs, sts, fes, ipanelfe, clus, pw, nlag, minhorz, subset,
        groups::Nothing, checkrows, TF=Float64)
    _checklpdata(ys, ws, nlag)
    Tfull = size(ys[1],1)
    # Number of rows possibly involved in estimation
    T = Tfull - nlag - minhorz
    nx = length(xs)
    nw = length(ws)
    nX = sts === nothing ? nx+nw*nlag : nx+nw*nlag*length(sts)
    T > nX || throw(ArgumentError(
        "not enough observations for nlag=$nlag and minhorz=$minhorz"))
    X = Matrix{TF}(undef, T, nX)
    if checkrows
        # Indicators for valid observations within the T rows
        esampleT = trues(T)
        # A cache for indicators
        aux = BitVector(undef, T)
        _fillX!(X, esampleT, aux, xs, ws, sts, fes, clus, pw, nlag, minhorz, subset, TF)
        sum(esampleT) > nX || throw(ArgumentError(
            "not enough observations for nlag=$nlag and minhorz=$minhorz"))
    else
        esampleT = nothing
        _fillX!(X, xs, ws, sts, nlag, minhorz, TF)
    end
    return LPData(ys, wgs, fes, ipanelfe, clus, pw, nlag, minhorz, subset, X,
        esampleT, groups)
end

function LPData(ys, xs, ws, wgs, sts, fes, ipanelfe, clus, pw, nlag, minhorz, subset,
        groups::Vector, checkrows, TF=Float64)
    _checklpdata(ys, ws, nlag)
    ng = length(groups)
    Tfull = size(ys[1],1)
    # Number of rows possibly involved in estimation
    T = Tfull - ng*nlag - ng*minhorz
    nx = length(xs)
    nw = length(ws)
    nX = sts === nothing ? nx+nw*nlag : nx+nw*nlag*length(sts)
    T > nX || throw(ArgumentError(
        "not enough observations for nlag=$nlag and minhorz=$minhorz"))
    X = Matrix{TF}(undef, T, nX)
    if checkrows
        # Indicators for valid observations within the T rows
        esampleT = trues(T)
        # A cache for indicators
        aux = BitVector(undef, T)
    else
        esampleT = nothing
    end
    r1 = 1
    for ids in groups
        Nid = length(ids)
        r2 = r1 + Nid - nlag - minhorz - 1
        gX = view(X, r1:r2, :)
        gxs = map(x->view(x, ids), xs)
        gws = map(x->view(x, ids), ws)
        gsts = sts === nothing ? nothing : map(x->view(x, ids), sts)
        gfes = map(x->view(x, ids), fes)
        gclus = map(x->view(x, ids), clus)
        gpw = pw === nothing ? nothing : view(pw, ids)
        if checkrows
            gesampleT = view(esampleT, r1:r2)
            gaux = view(aux, r1:r2)
            gsubset = subset === nothing ? nothing : view(subset, ids)
            _fillX!(gX, gesampleT, gaux, gxs, gws, gsts, gfes, gclus, gpw, nlag, minhorz, gsubset, TF)
        else
            _fillX!(gX, gxs, gws, gsts, nlag, minhorz, TF)
        end
        r1 = r2 + 1
    end
    if checkrows
        sum(esampleT) > nX || throw(ArgumentError(
            "not enough observations for nlag=$nlag and minhorz=$minhorz"))
    end
    return LPData(ys, wgs, fes, ipanelfe, clus, pw, nlag, minhorz, subset, X,
        esampleT, groups)
end

function _fillY!(Y, esampleT, aux, ys, nlag, horz, subset, isfirststage, TF)
    Tfull = size(ys[1], 1)
    for j in 1:size(Y,2)
        if isfirststage
            v = view(vec(ys[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
            subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull-horz))
        else
            v = view(vec(ys[j], subset, :y, horz, TF), nlag+horz+1:Tfull)
            subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+horz+1:Tfull))
        end
        _esample!(esampleT, aux, v)
        copyto!(view(Y,esampleT,j), view(v,esampleT))
    end
end

function _fillY!(Y, ys, nlag, horz, isfirststage, TF)
    Tfull = size(ys[1], 1)
    for j in 1:size(Y,2)
        if isfirststage
            v = view(vec(ys[j], nothing, :x, horz, TF), nlag+1:Tfull-horz)
        else
            v = view(vec(ys[j], nothing, :y, horz, TF), nlag+horz+1:Tfull)
        end
        copyto!(view(Y,:,j), v)
    end
end

# The simple case without the panel dimension
function _makeYX(dt::LPData{TF,Nothing}, horz::Int, isfirststage::Bool=false) where TF
    ny = length(dt.ys)
    # Number of rows possibly involved in estimation when horz==minhorz
    Tfull = size(dt.Xfull, 1)
    nshift = horz - dt.minhorz
    # Number of rows possibly involved in estimation for horz
    T = Tfull - nshift
    nX = size(dt.Xfull, 2)
    T > nX || throw(ArgumentError(
        "not enough observations for nlag=$nlag and horz=$horz"))
    Y = Matrix{TF}(undef, T, ny)
    if dt.esampleTfull isa BitVector
        # Indicators for valid observations within the T rows
        esampleT = dt.esampleTfull[1:T]
        # A cache for indicators
        aux = BitVector(undef, T)
        _fillY!(Y, esampleT, aux, dt.ys, dt.nlag, horz, dt.subset, isfirststage, TF)
        T1 = sum(esampleT)
        if T1 < T
            T1 > size(dt.Xfull,2) || throw(ArgumentError(
                "not enough observations for nlag=$nlag and horz=$(horz)"))
            Y = Y[esampleT, :]
        end
        X = Matrix{TF}(undef, T1, nX)
        @inbounds for j in 1:nX
            ir = 1
            for i in 1:T
                if esampleT[i]
                    X[ir,j] = dt.Xfull[i,j]
                    ir += 1
                end
            end
        end
        T = T1
    else
        esampleT = nothing
        _fillY!(Y, dt.ys, dt.nlag, horz, isfirststage, TF)
        X = dt.Xfull[1:T,:]
    end
    return Y, X, Y, nothing, nothing, nothing, uweights(T), T, esampleT, 0
end

# The case for panel data
# Need to additionally handle data for fixed effects, clusters and weights
# May also need to partial out wgs if relevant
function _makeYX(dt::LPData{TF,<:Vector}, horz::Int, isfirststage::Bool=false) where TF
    ny = length(dt.ys)
    # Number of rows possibly involved in estimation when horz==minhorz
    Tfull = size(dt.Xfull, 1)
    nshift = horz - dt.minhorz
    ng = length(dt.groups)
    # Number of rows possibly involved in estimation for horz
    T = Tfull - ng*nshift
    nX = size(dt.Xfull, 2)
    T > nX || throw(ArgumentError(
        "not enough observations for nlag=$nlag and horz=$horz"))
    Y = Matrix{TF}(undef, T, ny)
    checkrows = dt.esampleTfull isa BitVector
    if checkrows
        # Indicators for valid observations within the T rows
        esampleT = BitVector(undef, T)
        # A cache for indicators
        aux = BitVector(undef, T)
    else
        esampleT = nothing
    end
    r1 = 1
    i1 = 1
    for ids in dt.groups
        Nid = length(ids)
        step = Nid - dt.nlag - horz - 1
        r2 = r1 + step
        i2 = i1 + step
        gY = view(Y, i1:i2, :)
        gys = map(y->view(y, ids), dt.ys)
        if checkrows
            # Indicators for valid observations within the T rows
            gesampleT = view(esampleT, i1:i2)
            gesampleT .= dt.esampleTfull[r1:r2]
            gaux = view(aux, i1:i2)
            gsubset = dt.subset === nothing ? nothing : view(dt.subset, ids)
            _fillY!(gY, gesampleT, gaux, gys, dt.nlag, horz, gsubset, isfirststage, TF)
        else
            _fillY!(gY, gys, dt.nlag, horz, isfirststage, TF)
        end
        r1 = r2 + nshift + 1
        i1 = i2 + 1
    end
    if checkrows
        T1 = sum(esampleT)
        if T1 < T
            T1 > size(dt.Xfull,2) || throw(ArgumentError(
                "not enough observations for nlag=$nlag and horz=$(horz)"))
            Y = Y[esampleT, :]
        end
        X = Matrix{TF}(undef, T1, nX)
        @inbounds for j in 1:nX
            ir = 1
            i1 = 1
            k1 = 1
            for ids in dt.groups
                Nid = length(ids)
                step = Nid - dt.nlag - horz - 1
                i2 = i1 + step
                for i in i1:i2
                    if esampleT[k1]
                        X[ir,j] = dt.Xfull[i,j]
                        ir += 1
                    end
                    k1 += 1
                end
                i1 = i2 + nshift + 1
            end
        end
        FE = FixedEffect[]
        @inbounds for fe in dt.fes
            vfe = Vector{eltype(fe)}(undef, T1)
            ir = 1
            k1 = 1
            for ids in dt.groups
                i1 = ids[1] + dt.nlag
                i2 = ids[end] - horz
                for i in i1:i2
                    if esampleT[k1]
                        vfe[ir] = fe[i]
                        ir += 1
                    end
                    k1 += 1
                end
            end
            push!(FE, FixedEffect(vfe))
        end
        CLU = GroupedArray[]
        @inbounds for clu in dt.clus
            vclu = Vector{eltype(clu)}(undef, T1)
            ir = 1
            k1 = 1
            for ids in dt.groups
                i1 = ids[1] + dt.nlag
                i2 = ids[end] - horz
                for i in i1:i2
                    if esampleT[k1]
                        vclu[ir] = clu[i]
                        ir += 1
                    end
                    k1 += 1
                end
            end
            push!(CLU, GroupedArray(vclu, sort=nothing))
        end
        if dt.pw !== nothing
            W = Vector{TF}(undef, T1)
            ir = 1
            k1 = 1
            @inbounds for ids in dt.groups
                i1 = ids[1] + dt.nlag
                i2 = ids[end] - horz
                for i in i1:i2
                    if esampleT[k1]
                        W[ir] = dt.pw[i]
                        ir += 1
                    end
                    k1 += 1
                end
            end
            W = Weights(W)
        else
            W = uweights(T1)
        end
        T = T1
    else
        X = Matrix{TF}(undef, T, nX)
        @inbounds for j in 1:nX
            ir = 1
            i1 = 1
            for ids in dt.groups
                Nid = length(ids)
                step = Nid - dt.nlag - horz - 1
                i2 = i1 + step
                for i in i1:i2
                    X[ir,j] = dt.Xfull[i,j]
                    ir += 1
                end
                i1 = i2 + nshift + 1
            end
        end
        FE = FixedEffect[]
        @inbounds for fe in dt.fes
            vfe = Vector{eltype(fe)}(undef, T)
            ir = 1
            for ids in dt.groups
                i1 = ids[1] + dt.nlag
                i2 = ids[end] - horz
                for i in i1:i2
                    vfe[ir] = fe[i]
                    ir += 1
                end
            end
            push!(FE, FixedEffect(vfe))
        end
        CLU = GroupedArray[]
        @inbounds for clu in dt.clus
            vclu = Vector{eltype(clu)}(undef, T)
            ir = 1
            for ids in dt.groups
                i1 = ids[1] + dt.nlag
                i2 = ids[end] - horz
                for i in i1:i2
                    vclu[ir] = clu[i]
                    ir += 1
                end
            end
            push!(CLU, GroupedArray(vclu, sort=nothing))
        end
        if dt.pw !== nothing
            W = Vector{TF}(undef, T)
            ir = 1
            @inbounds for ids in dt.groups
                i1 = ids[1] + dt.nlag
                i2 = ids[end] - horz
                for i in i1:i2
                    W[ir] = dt.pw[i]
                    ir += 1
                end
            end
            W = Weights(W)
        else
            W = uweights(T)
        end
    end
    if isempty(FE)
        doffe = 0
    else
        doffe = 0
        for fe in FE
            if !isempty(CLU) && any(x->isnested(fe, x.groups), CLU)
                doffe += 1
            else
                doffe += nunique(fe)
            end
        end
        # Handle doffe via fe for intercepts added to wgs
        # but not residualize panelid via fe
        if isempty(dt.wgs) || dt.ipanelfe === nothing
            _feresiduals!(Y, X, FE, W)
        else
            if length(FE) > 1
                FE = FE[1:length(FE).!=dt.ipanelfe]
                _feresiduals!(Y, X, FE, W)
            end
        end
    end
    # Scaling of weights comes after partialing out fe as in FixedEffectModels.jl
    if !(W isa UnitWeights)
        Y .*= sqrt.(W)
        X .*= sqrt.(W)
    end
    if isempty(dt.wgs)
        pt = nothing
        Ypt = Y
    else
        Ypt = copy(Y)
        gT = length(dt.groups[1]) - dt.nlag - horz
        nwg = length(dt.wgs)
        # Must add a constant even with panelid as fixed effect
        gnx = nwg * dt.nlag + 1
        gny = ny + nX
        pt = PartialOLS(gT, gny, gnx, length(dt.groups), TF)
        pt.X[:,1] .= one(TF)
        for (g, ids) in enumerate(dt.groups)
            i1 = ids[1] + dt.nlag
            i2 = ids[end] - horz
            for j in 1:nwg
                for l in 1:dt.nlag
                    # No transformation for variables in wgs
                    v = view(dt.wgs[j], i1-l:i2-l)
                    # Variables with the same lag are put together
                    copyto!(view(pt.X,:,1+(l-1)*nwg+j), v)
                end
            end
            gY = view(Ypt, 1+(g-1)*gT:g*gT, :)
            copyto!(view(pt.resid, :, 1:ny), gY)
            gX = view(X, 1+(g-1)*gT:g*gT, :)
            copyto!(view(pt.resid, :, ny+1:gny), gX)
            if !(W isa UnitWeights)
                # pt.resid is already scaled by W
                pt.X[:,1] .= one(TF)
                pt.X .*= sqrt.(view(W, 1+(g-1)*gT:g*gT))
            end
            # Solve the partial OLS
            coefg = view(pt.coef, :, 1+(g-1)*gny:g*gny)
            mul!(coefg, pt.X', pt.resid)
            mul!(pt.crossX, pt.X', pt.X)
            ldiv!(cholesky!(pt.crossX), coefg)
            mul!(pt.resid, pt.X, coefg, -1.0, 1.0)
            # Replace the Y and X with residuals
            copyto!(gY, view(pt.resid, :, 1:ny))
            copyto!(gX, view(pt.resid, :, ny+1:gny))
        end
    end
    return Y, X, Ypt, pt, FE, CLU, W, T, esampleT, doffe
end

# Fitted values from first stage of 2SLS
function _fillfitted(fitted, groups::Nothing, nY, nlag, Tfull, horz, esampleT::BitVector, Xb)
    for i in 1:nY
        copyto!(view(view(fitted[i], nlag+1:Tfull-horz), esampleT), view(Xb,:,i))
    end
end

function _fillfitted(fitted, groups::Nothing, nY, nlag, Tfull, horz, esampleT::Nothing, Xb)
    for i in 1:nY
        copyto!(view(fitted[i], nlag+1:Tfull-horz), view(Xb,:,i))
    end
end

function _fillfitted(fitted, groups::Vector, nY, nlag, Tfull, horz, esampleT::BitVector, Xb)
    for i in 1:nY
        i1 = 1
        n1 = 1
        for ids in groups
            Nid = length(ids)
            step = Nid - nlag - horz - 1
            i2 = i1 + step
            gesampleT = view(esampleT, i1:i2)
            n2 = n1 + sum(gesampleT) - 1
            i1 = i2 + 1
            tar = view(fitted[i], ids)
            src = view(view(Xb,:,i), n1:n2)
            n1 = n2 + 1
            copyto!(view(view(tar, nlag+1:Nid-horz), gesampleT), src)
        end
    end
end

function _fillfitted(fitted, groups::Vector, nY, nlag, Tfull, horz, esampleT::Nothing, Xb)
    for i in 1:nY
        i1 = 1
        for ids in groups
            Nid = length(ids)
            step = Nid - nlag - horz - 1
            i2 = i1 + step
            tar = view(fitted[i], ids)
            src = view(view(Xb,:,i), i1:i2)
            i1 = i2 + 1
            copyto!(view(tar, nlag+1:Nid-horz), src)
        end
    end
end

# Xendo is needed for computing residuals from 2SLS
function _makeXendo(dt::LPData{TF,Nothing}, esampleT::BitVector, horz, yfs::Vector,
        FE, W) where TF
    nyfs = length(yfs)
    Xendo = Matrix{TF}(undef, sum(esampleT), nyfs)
    @inbounds for j in 1:nyfs
        yf = vec(yfs[j], dt.subset, :x, horz, TF)
        src = view(yf, dt.nlag+1:length(yf)-horz)
        copyto!(view(Xendo,:,j), view(src, esampleT))
    end
    W isa UnitWeights || (Xendo .*= sqrt.(W))
    return Xendo
end

function _makeXendo(dt::LPData{TF,Nothing}, esampleT::Nothing, horz, yfs::Vector,
        FE, W) where TF
    nyfs = length(yfs)
    T = size(dt.Xfull, 1) - horz + dt.minhorz
    Xendo = Matrix{TF}(undef, T, nyfs)
    @inbounds for j in 1:nyfs
        yf = vec(yfs[j], nothing, :x, horz, TF)
        src = view(yf, dt.nlag+1:length(yf)-horz)
        copyto!(view(Xendo,:,j), src)
    end
    W isa UnitWeights || (Xendo .*= sqrt.(W))
    return Xendo
end

function _makeXendo(dt::LPData{TF,<:Vector}, esampleT::BitVector, horz, yfs::Vector,
        FE, W) where TF
    nyfs = length(yfs)
    Xendo = Matrix{TF}(undef, sum(esampleT), nyfs)
    @inbounds for j in 1:nyfs
        yf = vec(yfs[j], dt.subset, :x, horz, TF)
        i1 = 1
        k1 = 1
        for ids in dt.groups
            Nid = length(ids)
            step = Nid - dt.nlag - horz - 1
            i2 = i1 + step
            gyf = view(yf, ids)
            src = view(gyf, dt.nlag+1:length(gyf)-horz)
            gesampleT = view(esampleT, i1:i2)
            k2 = k1 + sum(gesampleT) - 1
            copyto!(view(Xendo,k1:k2,j), view(src, gesampleT))
            i1 = i2 + 1
            k1 = k2 + 1
        end
    end
    _feresiduals!(Xendo, FE, W)
    W isa UnitWeights || (Xendo .*= sqrt.(W))
    return Xendo
end

function _makeXendo(dt::LPData{TF,<:Vector}, esampleT::Nothing, horz, yfs::Vector,
        FE, W) where TF
    nyfs = length(yfs)
    T = size(dt.Xfull, 1) - length(dt.groups) * (horz - dt.minhorz)
    Xendo = Matrix{TF}(undef, T, nyfs)
    @inbounds for j in 1:nyfs
        yf = vec(yfs[j], nothing, :x, horz, TF)
        i1 = 1
        for ids in dt.groups
            Nid = length(ids)
            step = Nid - dt.nlag - horz - 1
            i2 = i1 + step
            gyf = view(yf, ids)
            src = view(gyf, dt.nlag+1:length(gyf)-horz)
            copyto!(view(Xendo,i1:i2,j), src)
            i1 = i2 + 1
        end
    end
    _feresiduals!(Xendo, FE, W)
    W isa UnitWeights || (Xendo .*= sqrt.(W))
    return Xendo
end

show(io::IO, ::LPData) = print(io, "Data for local projection")
