@testset "OLS" begin
    T, N, K = 100, 2, 4
    X = randn(T, K)
    b = randn(K, N)
    Y = X * b
    m = OLS(Y, X, 10)
    @test m.coef ≈ b
    @test modelmatrix(m) === X
    @test coef(m) === m.coef
    @test residuals(m) === m.resid
    @test dof_residual(m) == 10
    @test dof_tstat(m) == dof_residual(m)
    m = OLS(Y, X, 10, 11)
    @test dof_tstat(m) == 11
    @test sprint(show, m) == "OLS regression"
end

@testset "LeastSquaresLP" begin
    e = LeastSquaresLP()
    @test sprint(show, e) == "LeastSquaresLP"
    @test sprint(show, MIME("text/plain"), e) == "Ordinary least squares local projection"
end

@testset "LeastSquaresLPResult" begin
    er = LeastSquaresLPResult([OLS(rand(5), rand(5,2), 5)])
    @test sprint(show, er) == "LeastSquaresLPResult"
end

@testset "LocalProjectionResult" begin
    B = cat(randn(5,2,1), randn(5,2,1); dims=3)
    V = cat(randn(10,10), randn(10,10); dims=3)
    r = LocalProjectionResult(B, V, [1,2], LeastSquaresLP(), nothing, HRVCE(), VarName[:y1, :y2], VarName[:x], VarName[:w1, :w2], VarName[], VarName[], Dict{VarName,Int}(:y1=>1, :y2=>2), Dict{VarName,Int}(:x=>1), Dict{VarName,Int}(:w1=>1,:w2=>2,:y1=>3,:y2=>4), nothing, nothing, 2, 0, nothing, nothing, nothing, nothing, nothing, nothing, false, nothing, nothing, true)
    @test coef(r, 1, :x) == B[1,1,1]
    @test coef(r, 2, :w1, yname=:y2, lag=1) == B[2,2,2]
    @test coef(r, 2, :w1, lag=2) == B[4,1,2]
    @test vcov(r, 1, :x) == V[1,1,1]
    @test vcov(r, 2, :x, yname1=:y2, xwname2=:w1, lag2=1) == V[2,4,2]
    @test vcov(r, 2, :x, yname1=:y2, xwname2=:w1, yname2=:y1, lag2=1) == V[2,3,2]
    @test vcov(r, 2, :w1, yname1=:y2, lag1=2, xwname2=:w1, yname2=:y1, lag2=1) == V[8,3,2]
    @test vcov(r, 2, :w2, yname1=:y2, lag1=2, xwname2=:w1, yname2=:y1) == V[10,7,2]

    @test sprint(show, r) == "LocalProjectionResult"
    @test sprint(show, MIME("text/plain"), r) == """
        LocalProjectionResult with 2 lags over 2 horizons:
        ──────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Outcome variables:              y1 y2    Minimum horizon:                    0
        Regressor:                          x    Lagged controls:                w1 w2
        ──────────────────────────────────────────────────────────────────────────────"""
    r = LocalProjectionResult(ones(1,1,1), ones(1,1,1), [1], LeastSquaresLP(), nothing, HRVCE(), VarName[:y], VarName[], VarName[:w], VarName[], VarName[], Dict{VarName,Int}(:y=>1), Dict{VarName,Int}(), Dict{VarName,Int}(:w=>1), nothing, nothing, 1, 0, nothing, nothing, nothing, nothing, nothing, nothing, false, nothing, nothing, true)
    @test sprint(show, MIME("text/plain"), r) == """
        LocalProjectionResult with 1 lag over 1 horizon:
        ──────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Outcome variable:                   y    Minimum horizon:                    0
        Regressor:                               Lagged control:                     w
        ──────────────────────────────────────────────────────────────────────────────"""
    r = LocalProjectionResult(ones(1,1,1), ones(1,1,1), [1], LeastSquaresLP(), nothing, HRVCE(), VarName[Cum(:y)], VarName[Cum(:x)], VarName[:w], VarName[:rec, :exp], VarName[], Dict{VarName,Int}(Cum(:y)=>1), Dict{VarName,Int}(Cum(:x)=>1), Dict{VarName,Int}(:w=>1), :pid, nothing, 1, 0, nothing, nothing, nothing, nothing, VarName[Cum(:x)], VarName[:z1,:z2], false, [0.12345678], 0.22345678, true)
    @test sprint(show, MIME("text/plain"), r) == """
        LocalProjectionResult with 1 lag over 1 horizon:
        ──────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Outcome variable:              Cum(y)    Minimum horizon:                    0
        Regressor:                     Cum(x)    Lagged control:                     w
        Endogenous variable:           Cum(x)    Instruments:                    z1 z2
        Kleibergen-Paap rk:     0.12 [0.2235]    States:                       rec exp
        ──────────────────────────────────────────────────────────────────────────────
        Panel Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Unit ID:                          pid    Weights:                    (uniform)
        Fixed effects:                 (none)    
        ──────────────────────────────────────────────────────────────────────────────"""

    r = LocalProjectionResult(B, V, [1,2], LeastSquaresLP(), nothing, HRVCE(), VarName[:y1, :y2], VarName[:x], VarName[:w1, :w2], VarName[], VarName[:fe1], Dict{VarName,Int}(:y1=>1, :y2=>2), Dict{VarName,Int}(:x=>1), Dict{VarName,Int}(:w1=>1,:w2=>2,:y1=>3,:y2=>4), :pid, :wt, 2, 0, nothing, nothing, nothing, nothing, nothing, nothing, false, nothing, nothing, true)
    @test sprint(show, MIME("text/plain"), r) == """
        LocalProjectionResult with 2 lags over 2 horizons:
        ──────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Outcome variables:              y1 y2    Minimum horizon:                    0
        Regressor:                          x    Lagged controls:                w1 w2
        ──────────────────────────────────────────────────────────────────────────────
        Panel Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Unit ID:                          pid    Weights:                           wt
        Fixed effects:                    fe1    
        ──────────────────────────────────────────────────────────────────────────────"""
end

@testset "lp" begin
    T = 100
    ys = Any[randn(T), randn(T)]
    xs = Any[randn(T), randn(T)]
    ws = ys
    dt = LPData(ys, xs, ws, nothing, Any[], Any[], nothing, 3, 0, nothing, nothing)
    b, v, t, m = _lp(dt, 5, SimpleVCE(), nothing, nothing)
    @test size(b) == (8, 2)
    @test size(v) == (16, 16)
    @test t == 92

    # Reproduce estimates from example based on Gertler & Karadi (2015)
    # Specification is non-augmented local projection
    # EHW standard errors
    # https://github.com/jm4474/Lag-augmented_LocalProjections/blob/master/examples/gk.m
    # The intercept estimated will be the same if not detrend the level before estimation
    # The left division in Matlab code uses QR factorization to solve OLS
    # The use of Cholesky factorization results in some trivial discrepancy
    df = exampledata(:gk)
    ns = (:logcpi, :logip, :ff, :ebp, :ff4_tc)
    r = lp(df, :ebp, wnames=ns, nlag=12, nhorz=48, vce=HRVCE())
    @test coef(r, 1, :ff4_tc, lag=1) ≈ 0.394494467439933 atol=1e-8
    @test coef(r, 5, :ff4_tc, lag=1) ≈ 0.147288229090455 atol=1e-8
    @test sqrt(vcov(r, 1, :ff4_tc, lag1=1)) ≈ 0.473834526724786 atol=1e-8
    @test sqrt(vcov(r, 5, :ff4_tc, lag1=1)) ≈ 0.737092370022356 atol=1e-8
    # Compare additional estimates obtained from Matlab lp before being turned into irf
    # Run in gk.m: lp(Y, 11, 1, 4, false, false)
    @test coef(r, 1, :constant) ≈ -2.945365821418727 atol=1e-8
    @test coef(r, 1, :logip, lag=1) ≈ -0.060094199554036 atol=1e-8
    @test vcov(r, 1, :constant) ≈ 0.641933721888132 atol=1e-8
    @test vcov(r, 1, :logcpi, lag1=1) ≈ 0.007331238288153 atol=1e-8
    @test vcov(r, 1, :logcpi, lag1=12) ≈ 0.004780116443880 atol=1e-8

    df[!,:gid] .= 1
    df[!,:wt] .= 2
    r1 = lp(df, :ebp, wnames=ns, nlag=12, nhorz=48, panelid=:gid, panelweight=:wt, vce=HRVCE())
    # Have one fewer coefficient because the constant term is replace by FE
    @test r1.B ≈ r.B[2:61,:,:]
    # V is not exactly the same because of the removed intercept
    @test r1.T ≈ r.T
    @test r1.fenames == [:gid]
    @test r1.panelid == :gid

    f = irf(r, :ebp, :ff4_tc, lag=1)
    @test coef(f)[1] ≈ 0.394494467439933 atol=1e-8
    @test coef(f)[20] ≈ -0.552849402428304 atol=1e-8
    @test stderror(f)[1] ≈ 0.473834526724786 atol=1e-8
    @test stderror(f)[20] ≈ 1.005046789996172 atol=1e-8
    ci = confint(f)
    @test ci[1][1] ≈ -0.3885764241953698 atol=1e-8
    @test ci[2][1] ≈ 1.1775653499255412 atol=1e-7
    @test ci[1][20] ≈ -2.2146531911738387 atol=1e-8
    @test ci[2][20] ≈ 1.1089543828008497 atol=1e-8

    ns1 = [3, :logip, 5, :ff4_tc]
    r1 = lp(df, 11, wnames=ns1, nlag=12, nhorz=1, vce=HRVCE())
    @test r1.B[1:3,1,1] ≈ r.B[1:3,1,1]
    @test r1.V[1:3,1:3,1] ≈ r.V[1:3,1:3,1]

    rn = lp(df, :ebp, xnames=:ff4_tc, wnames=(:ff4_tc,), nlag=12, nhorz=2,
        normalize=:ff4_tc=>:ff, vce=HRVCE())
    riv = lp(df, :ebp, xnames=:ff, wnames=(:ff4_tc,), nlag=12, nhorz=2,
        iv=:ff=>:ff4_tc, vce=HRVCE())
    @test coef(rn, 1, :ff4_tc) ≈ coef(riv, 1, :ff)
    @test coef(rn, 2, :ff4_tc) ≈ coef(riv, 2, :ff)
    @test rn.normnames == [:ff4_tc]
    @test rn.normtars == [:ff]
    @test rn.normmults ≈ [-5.3028242533068495] atol=1e-8
    @test riv.endonames == VarName[:ff]
    @test riv.ivnames == VarName[:ff4_tc]

    # Construct lags for comparing results with FixedEffectModels.jl
    for var in (:ebp, :ff4_tc)
        for l in 1:12
            df[!,Symbol(:l,l,var)] = vcat(fill(NaN, l), df[1:end-l, var])
        end
    end
    lebps = ntuple(i->term(Symbol(:l,i,:ebp)), 12)
    lff4_tc = ntuple(i->term(Symbol(:l,i,:ff4_tc)), 12)
    # This avoids issues on older versions of StatsModels.jl
    rhs = (term(:ff)~term(:ff4_tc), lebps..., lff4_tc...)
    rfe = reg(df, term(:ebp)~rhs, Vcov.robust())
    @test coef(riv, 1, :ff) ≈ coef(rfe)[end] atol=1e-8
    @test vcov(riv, 1, :ff) ≈ vcov(rfe)[end] atol=1e-8
    @test riv.F_kp ≈ rfe.F_kp atol=1e-8
    @test riv.p_kp ≈ rfe.p_kp atol=1e-8

    riv = lp(df, :ebp, xnames=:ff, wnames=(:ff4_tc,), nlag=12, nhorz=2,
        iv=:ff=>:ff4_tc, vce=HRVCE(), testweakiv=false)
    @test riv.F_kp === nothing
    @test riv.p_kp === nothing

    # Make sure that panel results with a single unit remain the same
    r1 = lp(df, :ebp, xnames=:ff4_tc, wnames=(:ff4_tc,), nlag=12, nhorz=2,
        normalize=:ff4_tc=>:ff, panelid=:gid, panelweight=:wt, vce=HRVCE())
    # Have one fewer coefficient because the constant term is replace by FE
    @test r1.B ≈ rn.B[(1:26).!=2,:,:]
    @test r1.T ≈ rn.T
    r1 = lp(df, :ebp, xnames=:ff, wnames=(:ff4_tc,), nlag=12, nhorz=2,
        iv=:ff=>:ff4_tc, panelid=:gid, panelweight=:wt, vce=HRVCE())
    @test r1.B ≈ riv.B[(1:26).!=2,:,:]
    @test r1.T ≈ riv.T

    @test_throws ArgumentError lp(df, :ebp, xnames=:ff4_tc, normalize=:no=>:ff)
    @test_throws ArgumentError lp(df, :ebp, xnames=:ff, iv=:no=>:ff4_tc)
    @test_throws ArgumentError lp(df, :ebp, xnames=:ff, iv=:ff=>())
    @test_throws ArgumentError lp(df, :ebp, xnames=:ff, iv=:ff=>1.0)

    # Compare standard errors based on EWC with Matlab results
    r2 = lp(r, HARVCE(EWC()))
    f2 = irf(r2, :ebp, :ff4_tc, lag=1)
    @test f2.B == f.B
    @test f2.T == f.T
    @test stderror(f2)[1] ≈ 0.408141982124863 atol=1e-8
    @test stderror(f2)[20] ≈ 1.037575254556690 atol=1e-8
    ci2 = confint(f2)
    @test ci2[1][1] ≈ -0.318073956754668 atol=1e-8
    @test ci2[2][1] ≈ 1.107062891634995 atol=1e-8
    @test ci2[1][20] ≈ -2.371771071486807 atol=1e-8
    @test ci2[2][20] ≈ 1.266072266630027 atol=1e-8

    # Reproduce point estimates in Table 1 from Ramey and Zubairy (2018)
    # Compare estimates produced by ivreg2 in Stata replication files
    # `jordagk.do` and `jordagk_twoinstruments.do`
    # Need to run `set type double` before executing the Stata do files for more precision
    # Use `bro h multlin1` to see the Stata results
    df = exampledata(:rz)

    # Baseline specifications
    r1 = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>:newsy,
        nlag=4, nhorz=17, addylag=false, firststagebyhorz=true, vce=HARVCE(EWC()))
    f1 = irf(r1, Cum(:y), Cum(:g))
    @test coef(f1)[1] ≈ 1.306459420753 atol=1e-8
    @test coef(f1)[9] ≈ .6689611146745 atol=1e-9
    @test coef(f1)[17] ≈ .7096110638625 atol=1e-9
    # Standard errors based on EWC rather than Newey-West in original paper
    @test stderror(f1)[1] ≈ 0.37439346775796917 atol=1e-8
    @test stderror(f1)[9] ≈ 0.05917848282306491 atol=1e-8
    @test stderror(f1)[17] ≈ 0.04441907325684153 atol=1e-8

    # Construct lags for comparing results with FixedEffectModels.jl
    # For rank test, heteroskedasticity-robust VCE is used even with vce=HARVCE(EWC())
    r1_0 = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>:newsy,
        nlag=4, nhorz=1, addylag=false, firststagebyhorz=true, vce=HARVCE(EWC()))
    r1_1 = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>:newsy,
        nlag=4, nhorz=1, addylag=false, firststagebyhorz=true, vce=HRVCE())
    for var in (:newsy, :y, :g)
        for l in 1:4
            df[!,Symbol(:l,l,var)] = vcat(fill(NaN, l), df[1:end-l, var])
        end
    end
    lnewsys = ntuple(i->term(Symbol(:l,i,:newsy)), 4)
    lys = ntuple(i->term(Symbol(:l,i,:y)), 4)
    lgs = ntuple(i->term(Symbol(:l,i,:g)), 4)
    rhs = (term(:g)~term(:newsy), lnewsys..., lys..., lgs...)
    rfe = reg(df, term(:y)~rhs, Vcov.robust())
    @test coef(r1_0, 1, Cum(:g)) ≈ coef(rfe)[end] atol=1e-8
    @test coef(r1_1, 1, Cum(:g)) ≈ coef(rfe)[end] atol=1e-8
    @test vcov(r1_1, 1, Cum(:g)) ≈ vcov(rfe)[end] atol=1e-8
    @test r1_0.F_kp[1] ≈ rfe.F_kp atol=1e-8
    @test r1_0.p_kp[1] ≈ rfe.p_kp atol=1e-8

    r2 = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>(:newsy, :g),
        nlag=4, nhorz=16, minhorz=1, addylag=false, firststagebyhorz=true, vce=HARVCE(EWC()))
    f2 = irf(r2, Cum(:y), Cum(:g))
    @test coef(f2)[1] ≈ .218045283661 atol=1e-9
    @test coef(f2)[8] ≈ .4509452247638 atol=1e-9
    @test coef(f2)[16] ≈ .5591002357465 atol=1e-9
    # Standard errors based on EWC rather than Newey-West in original paper
    @test stderror(f2)[1] ≈ 0.15265027120755809 atol=1e-8
    @test stderror(f2)[8] ≈ 0.0813792791970263 atol=1e-8
    @test stderror(f2)[16] ≈ 0.08518024610628232 atol=1e-8

    # Omit WWII
    r1 = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>:newsy,
        nlag=4, nhorz=17, addylag=false, firststagebyhorz=true, subset=df.wwii.==0)
    f1 = irf(r1, Cum(:y), Cum(:g))
    @test coef(f1)[1] ≈ 1.7188411171 atol=1e-8
    @test coef(f1)[9] ≈ .7672214331391 atol=1e-9
    @test coef(f1)[17] ≈ .7167179685971 atol=1e-9

    r2 = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>(:newsy, :g),
        nlag=4, nhorz=16, minhorz=1, addylag=false, firststagebyhorz=true, subset=df.wwii.==0)
    f2 = irf(r2, Cum(:y), Cum(:g))
    @test coef(f2)[1] ≈ -.0342369278527 atol=1e-10
    @test coef(f2)[8] ≈ .2570200212481 atol=1e-9
    @test coef(f2)[16] ≈ .2225592580744 atol=1e-9

    df[!,:gid] .= 1
    df[!,:wt] .= 2
    r3 = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>(:newsy, :g),
        nlag=4, nhorz=16, minhorz=1, addylag=false, firststagebyhorz=true, subset=df.wwii.==0,
        panelid=:gid, panelweight=:wt)
    @test r3.B ≈ r2.B[(1:14).!=2,:,:]

    # State dependency
    # Compare estimates generated from `jordagk_twoinstruments.do`
    # Use `bro multexpbh* multrecbh*` to see the Stata results

    # nomit
    r2 = lp(df, Cum(:y), xnames=(Cum(:g,:rec), Cum(:g,:exp), :rec), wnames=(:newsy, :y, :g),
        iv=(Cum(:g,:rec), Cum(:g,:exp))=>(:recnewsy, :expnewsy, :recg, :expg),
        states=(:rec, :exp), nlag=4, nhorz=16, minhorz=1, addylag=false, firststagebyhorz=true)
    f2rec = irf(r2, Cum(:y), Cum(:g,:rec))
    @test coef(f2rec)[1] ≈ .2718093668839 atol=1e-9
    @test coef(f2rec)[8] ≈ .6357587189621 atol=1e-9
    @test coef(f2rec)[16] ≈ .6784987089259 atol=1e-9
    f2exp = irf(r2, Cum(:y), Cum(:g,:exp))
    @test coef(f2exp)[1] ≈ .2660710337235 atol=1e-9
    @test coef(f2exp)[8] ≈ .3511868388287 atol=1e-9
    @test coef(f2exp)[16] ≈ .373442191223 atol=1e-9

    # wwii
    r2 = lp(df, Cum(:y), xnames=(Cum(:g,:rec), Cum(:g,:exp), :rec), wnames=(:newsy, :y, :g),
        iv=(Cum(:g,:rec), Cum(:g,:exp))=>(:recnewsy, :expnewsy, :recg, :expg),
        states=(:rec, :exp), nlag=4, nhorz=16, minhorz=1, addylag=false,
        firststagebyhorz=true, subset=df.wwii.==0)
    f2rec = irf(r2, Cum(:y), Cum(:g,:rec))
    @test coef(f2rec)[1] ≈ .345617820757 atol=1e-9
    @test coef(f2rec)[8] ≈ 1.350903038984 atol=1e-8
    @test coef(f2rec)[16] ≈ 1.365148452565 atol=1e-8
    f2exp = irf(r2, Cum(:y), Cum(:g,:exp))
    @test coef(f2exp)[1] ≈ .010365014606 atol=1e-10
    @test coef(f2exp)[8] ≈ .2171636455168 atol=1e-9
    @test coef(f2exp)[16] ≈ .2140385336746 atol=1e-9

    r21 = lp(df, Cum(:y), xnames=(Cum(:g,:rec), Cum(:g,:exp), :rec), wnames=(:newsy, :y, :g),
        iv=(Cum(:g,:rec), Cum(:g,:exp))=>(:recnewsy, :expnewsy, :recg, :expg),
        states=(:rec, :exp), nlag=4, nhorz=16, minhorz=1, addylag=false,
        firststagebyhorz=true, subset=df.wwii.==0, testweakiv=false)
    @test r21.B ≈ r2.B

    r3 = lp(df, Cum(:y), xnames=(Cum(:g,:rec), Cum(:g,:exp), :rec), wnames=(:newsy, :y, :g),
        iv=(Cum(:g,:rec), Cum(:g,:exp))=>(:recnewsy, :expnewsy, :recg, :expg),
        states=(:rec, :exp), nlag=4, nhorz=16, minhorz=1, addylag=false,
        firststagebyhorz=true, subset=df.wwii.==0, panelid=:gid)#, panelweight=:wt)
    @test r3.B ≈ r2.B[(1:28).!=4,:,:]

    @test_logs (:warn, "panelweight is ignored when panelid is nothing")
        lp(df, :y, xnames=:g, panelweight=:wt)
    @test_logs (:warn, "firststagebyhorz=false while endogenous variables contain Cum")
        lp(df, Cum(:y), xnames=Cum(:g), wnames=(:y,), iv=Cum(:g)=>:newsy, addylag=false)
    @test_logs (:warn, "addylag=true while outcome variables contain Cum")
        lp(df, Cum(:y), xnames=Cum(:g))

    # Compare Stata results from LP_example_panel.do posted on Jordà's website
    # Need `set type double` in Stata for precision
    df = exampledata(:jst)

    # Real GDP on STIR
    ws = (:dlgrgdp, :dlgcpi, :dstir)
    r = lp(df, :dlgrgdp, wnames=ws, nlag=3, nhorz=5, panelid=:iso)
    f = irf(r, :dlgrgdp, :dlgrgdp, lag=1)
    @test f.B[1] ≈ 0.2100997 atol=1e-7
    @test f.B[3] ≈ 0.0551249 atol=1e-7
    @test f.B[5] ≈ -.0762001 atol=1e-7
    f = irf(r, :dlgrgdp, :dstir, lag=1)
    @test f.B[2] ≈ -0.1599185 atol=1e-7
    @test f.B[4] ≈ -0.0047023 atol=1e-7

    r1 = lp(df, :dlgrgdp, wnames=ws, nlag=3, nhorz=1, panelid=:iso, vce=cluster(:iso))
    f1 = irf(r1, :dlgrgdp, :dlgrgdp, lag=1)
    # Construct lags for comparing confidence intervals with FixedEffectModels.jl
    lws = []
    for var in ws
        for l in 1:3
            v = Symbol(:l,l,var)
            f = x->lag(x, l)
            transform!(groupby(df, :iso), var=>f=>v)
            push!(lws, v)
        end
    end

    rhs = (term.(lws)..., fe(:iso))
    rfe = reg(df, term(:dlgrgdp)~rhs, cluster(:iso));
    @test f1.B[1] ≈ coef(rfe)[1] atol=1e-8
    @test stderror(f1)[1] ≈ stderror(rfe)[1] atol=1e-8
    @test dof_tstat(r1)[1] == 17
    f1ci = confint(f1, level=0.95)
    @test f1ci[1][1] ≈ confint(rfe)[1,1] atol=1e-8
    @test f1ci[2][1] ≈ confint(rfe)[1,2] atol=1e-8

    # Must fill in the missing values at the end even they are not used
    # This avoids reg from dropping additional rows at the end in the first-stage regression
    f = x->lag(x, -2, default=1.0)
    transform!(groupby(df,:iso), :dlgrgdp=>f=>:f2dlgrgdp)

    rhs = (term(:dlgrgdp)~term(:dlgcpi), term.(lws)..., fe(:iso))
    rfe = reg(df, term(:f2dlgrgdp)~rhs, cluster(:iso))
    r2 = lp(df, :f2dlgrgdp, xnames=:dlgrgdp, wnames=ws, nlag=3, nhorz=1,
        iv=:dlgrgdp=>:dlgcpi, panelid=:iso, vce=cluster(:iso), addylag=false)
    f2 = irf(r2, :f2dlgrgdp, :dlgrgdp)
    @test coef(r2, 1, :dlgrgdp) ≈ coef(rfe)[end] atol=1e-8
    @test stderror(f2)[1] ≈ stderror(rfe)[end] atol=1e-8
    @test r2.F_kp[1] ≈ rfe.F_kp atol=1e-8
    @test r2.p_kp[1] ≈ rfe.p_kp atol=1e-8
    @test dof_residual(r2)[1] == 2409

    r = lp(df, :dlgrgdp, wnames=ws, nlag=3, nhorz=5, panelid=:iso, addpanelidfe=false)
    @test isempty(r.fenames)

    @test_throws ArgumentError lp(df, :dlgrgdp, wnames=ws, fes=(:iso,))
    @test_throws ArgumentError lp(df, :dlgrgdp, wnames=ws, vce=cluster(:iso))
end
