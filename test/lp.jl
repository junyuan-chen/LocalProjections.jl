@testset "OLS" begin
    T, N, K = 100, 2, 4
    X = randn(T, K)
    b = randn(K, N)
    Y = X * b
    m = OLS(Y, X)
    @test m.coef ≈ b
    @test modelmatrix(m) === X
    @test coef(m) === m.coef
    @test residuals(m) === m.resid
    @test sprint(show, m) == "OLS regression"
end

@testset "LeastSquaresLP" begin
    e = LeastSquaresLP()
    @test sprint(show, e) == "LeastSquaresLP"
    @test sprint(show, MIME("text/plain"), e) == "Ordinary least squares local projection"
end

@testset "LeastSquaresLPResult" begin
    er = LeastSquaresLPResult([OLS(rand(5), rand(5,2))])
    @test sprint(show, er) == "LeastSquaresLPResult"
end

@testset "LocalProjectionResult" begin
    B = cat(randn(5,2,1), randn(5,2,1); dims=3)
    V = cat(randn(10,10), randn(10,10); dims=3)
    r = LocalProjectionResult(B, V, [1,2], LeastSquaresLP(), nothing, HRVCE(), VarName[:y1, :y2], VarName[:x], VarName[:w1, :w2], Dict{VarName,Int}(:y1=>1, :y2=>2), Dict{VarName,Int}(:x=>1), Dict{VarName,Int}(:w1=>1,:w2=>2,:y1=>3,:y2=>4), 2, 0, nothing, nothing, nothing, nothing, nothing, nothing, false, true)
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
    r = LocalProjectionResult(ones(1,1,1), ones(1,1,1), [1], LeastSquaresLP(), nothing, HRVCE(), VarName[:y], VarName[], VarName[:w], Dict{VarName,Int}(:y=>1), Dict{VarName,Int}(), Dict{VarName,Int}(:w=>1), 1, 0, nothing, nothing, nothing, nothing, nothing, nothing, false, true)
    @test sprint(show, MIME("text/plain"), r) == """
        LocalProjectionResult with 1 lag over 1 horizon:
        ──────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Outcome variable:                   y    Minimum horizon:                    0
        Regressor:                               Lagged control:                     w
        ──────────────────────────────────────────────────────────────────────────────"""
    r = LocalProjectionResult(ones(1,1,1), ones(1,1,1), [1], LeastSquaresLP(), nothing, HRVCE(), VarName[Cum(:y)], VarName[Cum(:x)], VarName[:w], Dict{VarName,Int}(Cum(:y)=>1), Dict{VarName,Int}(Cum(:x)=>1), Dict{VarName,Int}(:w=>1), 1, 0, nothing, nothing, nothing, nothing, VarName[Cum(:x)], VarName[:z1,:z2], false, true)
    @test sprint(show, MIME("text/plain"), r) == """
        LocalProjectionResult with 1 lag over 1 horizon:
        ──────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Outcome variable:              Cum(y)    Minimum horizon:                    0
        Regressor:                     Cum(x)    Lagged control:                     w
        Endogenous variable:           Cum(x)    Instruments:                    z1 z2
        ──────────────────────────────────────────────────────────────────────────────"""
end

@testset "_makeYX" begin
    T = 100
    ys = [randn(T), randn(T)]
    xs = [randn(T), randn(T)]
    ws = ys
    Y, X, T1, eT = _makeYX(ys, xs, ws, nothing, 3, 5, nothing)
    @test T1 == T-8
    @test size(Y) == (T1, 2)
    @test size(X) == (T1, 8)
    @test Y[:,1] == ys[1][9:end]
    @test Y[:,2] == ys[2][9:end]
    @test X[:,1] == xs[1][4:end-5]
    @test X[:,3] == ws[1][3:end-6]
    @test X[:,4] == ws[2][3:end-6]
    @test X[:,8] == ws[2][1:end-8]
    @test all(eT)

    sts = (rand(T), rand(T))
    Y, X, T1, eT = _makeYX(ys, xs, ws, sts, 1, 0, nothing)
    @test size(X) == (99, 6)
    @test X[:,1] == xs[1][2:end,1]
    @test X[:,2] == xs[2][2:end,1]
    @test X[:,3] == ys[1][1:99,1].*sts[1][2:end]
    @test X[:,4] == ys[1][1:99,1].*sts[2][2:end]
    @test X[:,5] == ys[2][1:99,1].*sts[1][2:end]
    @test X[:,6] == ys[2][1:99,1].*sts[2][2:end]
    @test all(eT)

    ys = (randn(T),)
    xs = ()
    ws = ys
    Y, X, T1, eT = _makeYX(ys, xs, ws, nothing, 1, 0, nothing)
    @test T1 == 99
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 1)
    @test Y == reshape(ys[1][2:end], T1, 1)
    @test X == reshape(ws[1][1:end-1], T1, 1)
    @test all(eT)

    ys[1][2], ys[1][3] = NaN, Inf
    xs = (convert(Vector{Union{Float64, Missing}}, randn(T)),)
    xs[1][3] = missing
    Y, X, T1, eT = _makeYX(ys, xs, ws, nothing, 1, 0, nothing)
    @test T1 == 96
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 2)
    @test Y == reshape(ys[1][5:end], T1, 1)
    @test X[:,1] == xs[1][5:end]
    @test X[:,2] == ws[1][4:end-1]
    @test eT == ((1:99).>3)

    Y, X, T1, eT = _makeYX(ys, xs, ws, nothing, 1, 1, (1:100).<=90)
    @test T1 == 85
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 2)
    @test Y == reshape(ys[1][6:90], T1, 1)
    @test X[:,1] == xs[1][5:89]
    @test X[:,2] == ws[1][4:88]
    # eT covers 2:99 in full data
    @test eT == ((1:98).>3) .& ((1:98).<=88)

    Y, X, T1, eT = _makeYX(ys, xs, ws, nothing, 1, 1, ((1:100).<60).|((1:100).>=70))
    @test T1 == 83
    @test Y[:] == vcat(ys[1][6:59], ys[1][72:end])
    @test X[:,1] == vcat(xs[1][5:58], xs[1][71:end-1])
    @test X[:,2] == vcat(ws[1][4:57], ws[1][70:end-2])
    @test eT == ((1:98).>3) .& (((1:98).<58) .| ((1:98).>=70))

    ys = ()
    @test_throws ArgumentError _makeYX(ys, xs, ws, nothing, 1, 0, nothing)
    ws = ()
    @test_throws ArgumentError _makeYX(ys, xs, ws, nothing, 1, 0, nothing)
    @test_throws ArgumentError _makeYX(ys, xs, ws, nothing, 0, 0, nothing)

    ys = ([Inf, NaN, 1.0],)
    xs = ()
    ws = ys
    @test_throws ArgumentError _makeYX(ys, xs, ws, nothing, 1, 0, nothing)
end

@testset "lp" begin
    T = 100
    ys = [randn(T), randn(T)]
    xs = [randn(T), randn(T)]
    ws = ys
    b, v, t, m = _lp(ys, xs, ws, nothing, 3, 5, SimpleVCE(), nothing)
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

    f = irf(r, :ebp, :ff4_tc, lag=1)
    @test coef(f)[1] ≈ 0.394494467439933 atol=1e-8
    @test coef(f)[20] ≈ -0.552849402428304 atol=1e-8
    @test stderror(f)[1] ≈ 0.473834526724786 atol=1e-8
    @test stderror(f)[20] ≈ 1.005046789996172 atol=1e-8
    ci = confint(f)
    @test ci[1][1] ≈ -0.384893972486403 atol=1e-8
    @test ci[2][1] ≈ 1.173882907366730 atol=1e-8
    @test ci[1][20] ≈ -2.206004261810359 atol=1e-8
    @test ci[2][20] ≈ 1.100305456953579 atol=1e-8

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
    @test vcov(rn, 1, :ff4_tc) ≈ vcov(riv, 1, :ff)
    @test vcov(rn, 2, :ff4_tc) ≈ vcov(riv, 2, :ff)
    @test rn.normnames == [:ff4_tc]
    @test rn.normtars == [:ff]
    @test rn.normmults ≈ [-5.3028242533068495] atol=1e-8
    @test riv.endonames == VarName[:ff]
    @test riv.ivnames == VarName[:ff4_tc]

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
        nlag=4, nhorz=17, addylag=false, firststagebyhorz=true)
    f1 = irf(r1, Cum(:y), Cum(:g))
    @test coef(f1)[1] ≈ 1.306459420753 atol=1e-8
    @test coef(f1)[9] ≈ .6689611146745 atol=1e-9
    @test coef(f1)[17] ≈ .7096110638625 atol=1e-9
    # Standard errors based on EWC rather than Newey-West in original paper
    @test stderror(f1)[1] ≈ 0.4286854811143802 atol=1e-8
    @test stderror(f1)[9] ≈ 0.10921291172237432 atol=1e-8
    @test stderror(f1)[17] ≈ 0.20049776900175723 atol=1e-8

    r2 = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>(:newsy, :g),
        nlag=4, nhorz=16, minhorz=1, addylag=false, firststagebyhorz=true)
    f2 = irf(r2, Cum(:y), Cum(:g))
    @test coef(f2)[1] ≈ .218045283661 atol=1e-9
    @test coef(f2)[8] ≈ .4509452247638 atol=1e-9
    @test coef(f2)[16] ≈ .5591002357465 atol=1e-9
    # Standard errors based on EWC rather than Newey-West in original paper
    @test stderror(f2)[1] ≈ 0.16423131478595118 atol=1e-8
    @test stderror(f2)[8] ≈ 0.10754779358937377 atol=1e-8
    @test stderror(f2)[16] ≈ 0.14263798115850218 atol=1e-8

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

    # State dependency
    # Compare estimates generated from `jordagk_twoinstruments.do`
    # Use `bro multexpbh* multrecbh*` to see the Stata results

    # Construct the lag of slack
    df.rec = similar(df.slack)
    df.rec[1] = missing
    # Fill the beginning missing values
    df.rec[2:5] .= true
    df.rec[6:end] .= view(df.slack, 5:length(df.rec)-1)
    df.exp = 1.0 .- df.rec
    df.recnewsy = df.rec.*df.newsy
    df.expnewsy = df.exp.*df.newsy
    df.recg = df.rec.*df.g
    df.expg = df.exp.*df.g

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
        states=(:rec, :exp), nlag=4, nhorz=16, minhorz=1, addylag=false, firststagebyhorz=true,
        subset=df.wwii.==0)
    f2rec = irf(r2, Cum(:y), Cum(:g,:rec))
    @test coef(f2rec)[1] ≈ .345617820757 atol=1e-9
    @test coef(f2rec)[8] ≈ 1.350903038984 atol=1e-8
    @test coef(f2rec)[16] ≈ 1.365148452565 atol=1e-8
    f2exp = irf(r2, Cum(:y), Cum(:g,:exp))
    @test coef(f2exp)[1] ≈ .010365014606 atol=1e-10
    @test coef(f2exp)[8] ≈ .2171636455168 atol=1e-9
    @test coef(f2exp)[16] ≈ .2140385336746 atol=1e-9

    @test_logs (:warn, "firststagebyhorz=false while endogenous variables contain Cum")
        lp(df, Cum(:y), xnames=Cum(:g), wnames=(:y,), iv=Cum(:g)=>:newsy, addylag=false)
    @test_logs (:warn, "addylag=true while outcome variables contain Cum")
        lp(df, Cum(:y), xnames=Cum(:g))
end
