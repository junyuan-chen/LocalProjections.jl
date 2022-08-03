"""
    LPData{TF<:AbstractFloat,TG<:Union{Vector,Nothing},TW<:Union{AbstractVector,Nothing}}

Raw data collected for generating data matrices for local projection estimation.
"""
struct LPData{TF<:AbstractFloat,TG<:Union{Vector,Nothing},TW<:Union{AbstractVector,Nothing}}
    ys::Vector{Any}
    xs::Vector{Any}
    ws::Vector{Any}
    sts::Union{Vector{Any},Nothing}
    fes::Vector{Any}
    clus::Vector{Any}
    pw::TW
    nlag::Int
    minhorz::Int
    subset::Union{BitVector,Nothing}
    Xfull::Matrix{TF}
    esampleTfull::BitVector
    groups::TG
end

function _fillX!(X, esampleT, aux, xs, ws, sts, fes, clus, pw, nlag, horz, subset, TF)
    Tfull = size(ws[1],1)
    nx = length(xs)
    nw = length(ws)
    nfe = length(fes)
    nclu = length(clus)
    if nx > 0
        for j in 1:nx
            v = view(vec(xs[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
            subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull-horz))
            _esample!(esampleT, aux, v)
            copyto!(view(X,esampleT,j), view(v,esampleT))
        end
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
        ssts = Vector{AbstractVector}(undef, ns)
        for j in 1:ns
            v = view(vec(sts[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
            _esample!(esampleT, aux, v)
            ssts[j] = v
        end
        for j in 1:nw
            for l in 1:nlag
                for s in 1:ns
                    v = view(vec(ws[j], subset, :x, horz, TF), nlag+1-l:Tfull-horz-l)
                    subset === nothing || j > 1 ||
                        (esampleT .&= view(subset, nlag+1-l:Tfull-horz-l))
                    s > 1 || _esample!(esampleT, aux, v)
                    X[esampleT, nx+(l-1)*nw*ns+(j-1)*ns+s] .=
                        view(ssts[s],esampleT) .* view(v,esampleT)
                end
            end
        end
    end
    if nfe > 0
        for j in 1:nfe
            v = view(vec(fes[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
            subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull-horz))
            _esample!(esampleT, aux, v)
        end
    end
    if nclu > 0
        for j in 1:nclu
            v = view(vec(clus[j], subset, :x, horz, TF), nlag+1:Tfull-horz)
            subset === nothing || j > 1 || (esampleT .&= view(subset, nlag+1:Tfull-horz))
            _esample!(esampleT, aux, v)
        end
    end
    if pw !== nothing
        v = view(vec(pw, subset, :x, horz, TF), nlag+1:Tfull-horz)
        subset === nothing || (esampleT .&= view(subset, nlag+1:Tfull-horz))
        _esample!(esampleT, aux, v)
    end
end

function _checklpdata(ys, ws, nlag)
    length(ys) > 0 || throw(ArgumentError("ys cannot be empty"))
    nlag > 0 || throw(ArgumentError("nlag must be at least 1"))
    length(ws) > 0 || throw(ArgumentError("ws cannot be empty"))
end

function LPData(ys, xs, ws, sts, fes, clus, pw, nlag, minhorz, subset,
        groups::Nothing, TF=Float64)
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
    # Indicators for valid observations within the T rows
    esampleT = trues(T)
    # A cache for indicators
    aux = BitVector(undef, T)
    _fillX!(X, esampleT, aux, xs, ws, sts, fes, clus, pw, nlag, minhorz, subset, TF)
    sum(esampleT) > nX || throw(ArgumentError(
        "not enough observations for nlag=$nlag and minhorz=$minhorz"))
    return LPData(ys, xs, ws, sts, fes, clus, pw, nlag, minhorz, subset, X, esampleT, groups)
end

function LPData(ys, xs, ws, sts, fes, clus, pw, nlag, minhorz, subset,
        groups::Vector, TF=Float64)
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
    # Indicators for valid observations within the T rows
    esampleT = trues(T)
    # A cache for indicators
    aux = BitVector(undef, T)
    r1 = 1
    for ids in groups
        Nid = length(ids)
        r2 = r1 + Nid - nlag - minhorz - 1
        gX = view(X, r1:r2, :)
        gesampleT = view(esampleT, r1:r2)
        gaux = view(aux, r1:r2)
        r1 = r2 + 1
        gxs = map(x->view(x, ids), xs)
        gws = map(x->view(x, ids), ws)
        gsts = sts === nothing ? nothing : map(x->view(x, ids), sts)
        gfes = map(x->view(x, ids), fes)
        gclus = map(x->view(x, ids), clus)
        gpw = pw === nothing ? nothing : view(pw, ids)
        gsubset = subset === nothing ? nothing : view(subset, ids)
        _fillX!(gX, gesampleT, gaux, gxs, gws, gsts, gfes, gclus, gpw, nlag, minhorz, gsubset, TF)
    end
    sum(esampleT) > nX || throw(ArgumentError(
        "not enough observations for nlag=$nlag and minhorz=$minhorz"))
    return LPData(ys, xs, ws, sts, fes, clus, pw, nlag, minhorz, subset, X, esampleT, groups)
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

# The simple case without the panel dimension
function _makeYX(dt::LPData{TF,Nothing}, horz::Int, isfirststage::Bool=false) where TF
    ny = length(dt.ys)
    # Number of rows possibly involved in estimation when horz==minhorz
    Tfull = length(dt.esampleTfull)
    nshift = horz - dt.minhorz
    # Number of rows possibly involved in estimation for horz
    T = Tfull - nshift
    nX = size(dt.Xfull, 2)
    T > nX || throw(ArgumentError(
        "not enough observations for nlag=$nlag and horz=$horz"))
    # Indicators for valid observations within the T rows
    esampleT = dt.esampleTfull[1:T]
    # A cache for indicators
    aux = BitVector(undef, T)
    # Construct matrices
    Y = Matrix{TF}(undef, T, ny)
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
    return Y, X, nothing, uweights(T), T, esampleT, 0
end

# The case for panel data
# Need to additionally handle data for fixed effects, clusters and weights
function _makeYX(dt::LPData{TF,<:Vector}, horz::Int, isfirststage::Bool=false) where TF
    ny = length(dt.ys)
    # Number of rows possibly involved in estimation when horz==minhorz
    Tfull = length(dt.esampleTfull)
    nshift = horz - dt.minhorz
    ng = length(dt.groups)
    # Number of rows possibly involved in estimation for horz
    T = Tfull - ng*nshift
    nX = size(dt.Xfull, 2)
    T > nX || throw(ArgumentError(
        "not enough observations for nlag=$nlag and horz=$horz"))
    # Indicators for valid observations within the T rows
    esampleT = BitVector(undef, T)
    # A cache for indicators
    aux = BitVector(undef, T)
    # Construct matrices
    Y = Matrix{TF}(undef, T, ny)
    r1 = 1
    i1 = 1
    for ids in dt.groups
        Nid = length(ids)
        step = Nid - dt.nlag - horz - 1
        r2 = r1 + step
        i2 = i1 + step
        # Indicators for valid observations within the T rows
        gesampleT = view(esampleT, i1:i2)
        gesampleT .= dt.esampleTfull[r1:r2]
        gaux = view(aux, i1:i2)
        gY = view(Y, i1:i2, :)
        r1 = r2 + nshift + 1
        i1 = i2 + 1
        gys = map(y->view(y, ids), dt.ys)
        gsubset = dt.subset === nothing ? nothing : view(dt.subset, ids)
        _fillY!(gY, gesampleT, gaux, gys, dt.nlag, horz, gsubset, isfirststage, TF)
    end
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
    if isempty(FE)
        doffe = 0
    else
        _feresiduals!(Y, X, FE, W)
        doffe = 0
        for fe in FE
            if !isempty(CLU) && any(x->isnested(fe, x.groups), CLU)
                doffe += 1
            else
                doffe += nunique(fe)
            end
        end
    end
    if !(W isa UnitWeights)
        for col in eachcol(Y)
            col .*= sqrt.(W)
        end
        for col in eachcol(X)
            col .*= sqrt.(W)
        end
    end
    T = T1
    return Y, X, CLU, W, T, esampleT, doffe
end

# Xendo is needed for computing residuals from 2SLS
function _makeXendo(dt::LPData{TF,Nothing}, esampleT::BitVector, horz, yfs::Vector) where TF
    nyfs = length(yfs)
    Xendo = Matrix{TF}(undef, sum(esampleT), nyfs)
    @inbounds for j in 1:nyfs
        yf = vec(yfs[j], dt.subset, :x, horz, TF)
        src = view(yf, dt.nlag+1:length(yf)-horz)
        copyto!(view(Xendo,:,j), view(src, esampleT))
    end
    return Xendo
end

function _makeXendo(dt::LPData{TF,<:Vector}, esampleT::BitVector, horz, yfs::Vector) where TF
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
    return Xendo
end

show(io::IO, ::LPData) = print(io, "Data for local projection")
