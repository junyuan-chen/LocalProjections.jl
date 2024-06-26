@testset "Ridge" begin
    a2 = rand(2,2)
    C = rand(5,2)
    θ = rand(2)
    resid = rand(5)
    m = Ridge(rand(5), C, a2, rand(2), a2, 1.0, a2, nothing, 1.0, θ, resid, a2, 1.0, 1)
    @test modelmatrix(m) === C
    @test coef(m) === θ
    @test residuals(m) === resid
    @test sprint(show, m) == "Ridge regression"
end

@testset "SearchCriterion" begin
    c = LOOCV()
    @test sprint(show, c) == "LOOCV"
    @test sprint(show, MIME("text/plain"), c) == "Leave-one-out cross validation"

    c = GCV()
    @test sprint(show, c) == "GCV"
    @test sprint(show, MIME("text/plain"), c) == "Generalized cross validation"

    c = AIC()
    @test sprint(show, c) == "AIC"
    @test sprint(show, MIME("text/plain"), c) == "Akaike information criterion"
end

@testset "SmoothAlgorithm" begin
    a = DemmlerReinsch()
    @test sprint(show, a) == "DemmlerReinsch"
    @test sprint(show, MIME("text/plain"), a) == "Demmler-Reinsch orthogonalization"

    a = DirectSolve()
    @test sprint(show, a) == "DirectSolve"
end

@testset "GridSearch" begin
    @test grid(1) == GridSearch([1.0])
    @test grid(1:3) == GridSearch([1.0, 2.0, 3.0])
    @test grid([2.0, 1]) == GridSearch([2.0, 1.0])
    @test grid().v == exp.(LinRange(-6, 12, 50))
    @test grid([2 1], algo=DirectSolve()) == GridSearch([2.0, 1.0], algo=DirectSolve())
    @test_throws ArgumentError grid([2.0, -1.0])

    s = grid()
    @test sprint(show, s) == "GridSearch"
    @test sprint(show, MIME("text/plain"), s)[1:80] == """
        GridSearch{DemmlerReinsch, Float64} across 50 candidate values:
          [0.00247875, 0"""
end

@testset "GridSearchResult" begin
    iopt = Dict{SearchCriterion,Int}()
    iopt[LOOCV()] = 5
    iopt[GCV()] = 1
    iopt[AIC()] = 3
    v = rand(2)
    sr = GridSearchResult(iopt, rand(2,2), v, v, v, v, v, v)
    @test sprint(show, sr) == "GridSearchResult"
    @test sprint(show, MIME("text/plain"), sr) == """
        GridSearchResult across 2 candidate values:
          LOOCV => 5
          GCV   => 1
          AIC   => 3"""
end

@testset "SmoothLP" begin
    e = SmoothLP(:v)
    @test sprint(show, e) == "SmoothLP"
    @test sprint(show, MIME("text/plain"), e) == """
        Smooth Local Projection:
          smoothed regressor: v
          polynomial order: 3
          finite difference order: 3"""
    e = SmoothLP([:v1, 2], 2, 2, search=grid(), criterion=AIC(), fullX=true)
    @test sprint(show, MIME("text/plain"), e) == """
        Smooth Local Projection:
          smoothed regressors: v1 2
          polynomial order: 2
          finite difference order: 2"""
end

@testset "_basismatrix" begin
    b3 = [1/6 2/3 1/6]
    @test _basismatrix(3, 0, 0) ≈ b3
    bm = _basismatrix(3, 0, 2)
    @test size(bm) == (3,5)
    m = zeros(3, 5)
    for r in 1:3
        m[r,r:r+2] = b3
    end
    @test bm ≈ m
    @test _basismatrix(2, 0, 0) ≈ [0.5 0.5]
end

@testset "_makeYSr" begin
    T = 100
    ys, ss, xs = Any[randn(T)], [randn(T)], Any[randn(T)]
    ws = ys
    wgs = Any[]
    fes0 = Any[]
    clus0 = Any[]
    dt = LPData(ys, xs, ws, wgs, nothing, fes0, nothing, clus0, nothing, 3, 0,
        nothing, nothing, true)
    res, X, T1, e1, e2 = _makeYSr(dt, ss, 5)
    @test T1 == T-8
    @test size(res) == (T1, 2)
    @test all(e1)
    @test all(e2)

    xs = Any[]
    dt = LPData(ys, xs, ws, wgs, nothing, fes0, nothing, clus0, nothing, 1, 0,
        nothing, nothing, true)
    res, X, T1, e1, e2 = _makeYSr(dt, ss, 0)
    @test T1 == 99
    @test size(res) == (T1, 2)
    @test all(e1)
    @test all(e2)

    ys[1][2], ys[1][3] = NaN, Inf
    xs = Any[convert(Vector{Union{Float64, Missing}}, randn(T))]
    xs[1][3] = missing
    dt = LPData(ys, xs, ws, wgs, nothing, fes0, nothing, clus0, nothing, 1, 0,
        nothing, nothing, true)
    res, X, T1, e1, e2 = _makeYSr(dt, ss, 0)
    @test T1 == 96
    @test size(res) == (T1, 2)
    @test e1 == ((1:99).>3)
    @test all(e2)

    dt = LPData(ys, xs, ws, wgs, nothing, fes0, nothing, clus0, nothing, 1, 1,
        ((1:100).<60).|((1:100).>=70), nothing, true)
    res, X, T1, e1, e2 = _makeYSr(dt, ss, 1)
    @test T1 == 83
    @test e1 == ((1:98).>3) .& (((1:98).<58) .| ((1:98).>=70))
    @test all(e2)

    ys, xs = Any[randn(T)], Any[randn(T)]
    ss[1][2], ss[1][4] = NaN, Inf
    ws = ys
    dt = LPData(ys, xs, ws, wgs, nothing, fes0, nothing, clus0, nothing, 2, 0,
        nothing, nothing, true)
    res, X, T1, e1, e2 = _makeYSr(dt, ss, 0)
    @test T1 == 97
    @test all(e1)
    @test e2 == ((1:98).!=2)
end

@testset "slp" begin
    # Check estimates are close to OLS with low penalty
    df = exampledata(:gk)
    ns = (:logcpi, :logip, :ff, :ebp, :ff4_tc)
    r0 = lp(df, :ebp, xnames=:ff4_tc, wnames=ns, nlag=12, nhorz=48, vce=HRVCE())
    f0 = irf(r0, :ebp, :ff4_tc)
    est = SmoothLP(:ff4_tc, search=grid(1e-8), criterion=GCV())
    r1 = lp(est, df, :ebp, xnames=:ff4_tc, wnames=ns, nlag=12, nhorz=48, vce=HRVCE())
    f1 = irf(r1, :ebp, :ff4_tc)
    @test f1.B ≈ f0.B atol=1e-4
    # Check HR confidence interval
    ci1 = confint(f1)
    @test ci1[1][1] ≈ -0.12201693146448783 atol=1e-8
    @test ci1[2][1] ≈ 0.9502829017743402 atol=1e-8
    @test ci1[1][10] ≈ -0.5694889939392841 atol=1e-8
    @test ci1[2][10] ≈ 2.4773514368646214 atol=1e-8
    @test r1.estres.m.dof_adj == 48*61
    # With λ=1e-8, the dofr is smaller than that for least-squares LP (8280)
    @test dof_residual(r1) ≈ 8233.90911815367 atol=1e-6
    @test dof_tstat(r1) == dof_residual(r1)

    @test sprint(show, MIME("text/plain"), r1) == """
        LocalProjectionResult with 12 lags over 48 horizons:
        ──────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Outcome variable:                 ebp    Minimum horizon:                    0
        Regressors:           ff4_tc constant    
        Lagged controls:                                    logcpi logip ff ebp ff4_tc
        ──────────────────────────────────────────────────────────────────────────────
        Smooth Local Projection
        ──────────────────────────────────────────────────────────────────────────────
        Smoothing parameter:             0.00    Smoothed regressor:            ff4_tc
        Polynomial order:                   3    Finite difference order:            3
        Selection criterion:              GCV    Selection algorithm:   DemmlerReinsch
        Leave-one-out CV:             2212.02    Generalized CV:               2212.25
        Akaike information:              7.70    Residual sum of squares:      2193.42
        ──────────────────────────────────────────────────────────────────────────────"""

    # Compare estimates with Matlab results from Barnichon and Brownlees (2019)
    df = exampledata(:bb)
    ns = (:ir, :pi, :yg)
    # 194 is their optimal smoothing parameter
    est = SmoothLP(:ir, 3, 2, search=grid(194), criterion=LOOCV())
    r1 = lp(est, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1, vce=HARVCE(EWC()))
    # δ is the normalization factor they use taken from Matlab
    δ = 0.802327024988408
    f1 = irf(r1, :yg, :ir)
    B1 = f1.B.*δ
    @test B1[1] ≈ -0.189377969344491 atol=1e-10
    @test B1[2] ≈ -0.429648467594006 atol=1e-10
    @test B1[3] ≈ -0.533856512472860 atol=1e-10
    @test B1[9] ≈ -0.022406955605936 atol=1e-10
    @test B1[10] ≈ 0.146703133067349 atol=1e-10
    @test B1[11] ≈ 0.303602580252881 atol=1e-10
    @test B1[18] ≈ 0.222464247242188 atol=1e-10
    @test B1[19] ≈ 0.096090022750258 atol=1e-10
    @test B1[20] ≈ -0.065709407174569 atol=1e-10

    # Check EWC confidence interval
    ci1 = confint(f1)
    @test ci1[1][1] ≈ -0.8830584463829632 atol=1e-8
    @test ci1[2][1] ≈ 0.4109866765273089 atol=1e-8
    @test ci1[1][10] ≈ 0.013380390754119786 atol=1e-8
    @test ci1[2][10] ≈ 0.3523137177534175 atol=1e-8

    # Compare results based on DemmlerReinsch and DirectSolve
    gbb = 194.0.*(1:0.5:10)
    iopt = Dict{SearchCriterion,Int}(c=>1 for c in (LOOCV(), GCV(), AIC()))
    est1 = SmoothLP(:ir, 3, 2, search=grid(gbb), criterion=LOOCV())
    r1 = lp(est1, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1)
    @test r1.estres.λ == 194
    @test r1.estres.search.iopt == iopt
    @test r1.estres.rss ≈ sum(r1.estres.m.resid.^2)
    est2 = SmoothLP(:ir, 3, 2, search=grid(gbb, algo=DirectSolve()), criterion=GCV())
    r2 = lp(est2, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1)
    @test r2.estres.λ == 194
    @test r2.estres.search.iopt == iopt
    @test r2.estres.rss ≈ sum(r2.estres.m.resid.^2)
    @test r1.estres.search.θs ≈ r2.estres.search.θs atol=1e-2
    @test r1.estres.search.loocv ≈ r2.estres.search.loocv atol=1e-1
    @test r1.estres.search.rss ≈ r2.estres.search.rss atol=1e-1
    @test r1.estres.search.gcv ≈ r2.estres.search.gcv atol=1e-1
    @test r1.estres.search.aic ≈ r2.estres.search.aic atol=1e-5
    @test r1.estres.search.dof_fit ≈ r2.estres.search.dof_fit atol=1e-2
    # Check the recalculated final estimates are the same
    @test r1.estres.θ ≈ r2.estres.θ
    @test r1.estres.Σ ≈ r2.estres.Σ
    @test r1.estres.loocv ≈ r2.estres.loocv
    @test r1.estres.rss ≈ r2.estres.rss
    @test r1.estres.gcv ≈ r2.estres.gcv
    @test r1.estres.aic ≈ r2.estres.aic
    @test r1.estres.dof_fit ≈ r2.estres.dof_fit
    @test r1.estres.dof_res ≈ r2.estres.dof_res

    # Compare estimates from LeastSquaresLP
    r0 = lp(df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1)
    f0 = irf(r0, :yg, :ir)
    est3 = SmoothLP(:ir, 3, 2, search=grid(1e-8), criterion=AIC())
    r3 = lp(est3, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1)
    f3 = irf(r3, :yg, :ir)
    @test f3.B ≈ f0.B atol=1e-8

    # Make a benchmark for comparisons
    est0 = SmoothLP(:ir, 3, 2, search=grid(194.0))
    r0 = lp(est0, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1, vce=HRVCE())
    r4 = lp(r1, HRVCE())
    @test r4.B ≈ r1.B
    @test r4.V ≈ r0.V
    f4 = irf(r4, :yg, :ir)
    ci4 = confint(f4)
    @test ci4[1][1] ≈ -0.6976347030646475 atol=1e-8
    @test ci4[2][1] ≈ 0.22556293320899323 atol=1e-8
    @test ci4[1][10] ≈ -0.07572169749443655 atol=1e-8
    @test ci4[2][10] ≈ 0.44141580600197383 atol=1e-8

    r5 = lp(r1, 194.0, vce=HRVCE())
    @test r5.B ≈ r1.B
    @test r5.V ≈ r0.V
    est0 = SmoothLP(:ir, 3, 2, search=grid(30))
    r0 = lp(est0, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1, vce=HRVCE())
    r6 = lp(r1, 30, vce=HRVCE())
    @test r6.B ≈ r0.B
    @test r6.V ≈ r0.V

    @test sprint(show, MIME("text/plain"), r1) == """
        LocalProjectionResult with 4 lags over 20 horizons:
        ──────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Outcome variable:                  yg    Minimum horizon:                    1
        Regressors:         ir pi yg constant    Lagged controls:             ir pi yg
        ──────────────────────────────────────────────────────────────────────────────
        Smooth Local Projection
        ──────────────────────────────────────────────────────────────────────────────
        Smoothing parameter:           194.00    Smoothed regressor:                ir
        Polynomial order:                   3    Finite difference order:            2
        Selection criterion:            LOOCV    Selection algorithm:   DemmlerReinsch
        Leave-one-out CV:            34530.21    Generalized CV:              34511.09
        Akaike information:             10.45    Residual sum of squares:     34379.97
        ──────────────────────────────────────────────────────────────────────────────"""

    df = exampledata(:rz)
    est = SmoothLP(Cum(:g), 3, 3, search=grid([1, 1e3, 1e5]))
    r1 = lp(est, df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g), iv=Cum(:g)=>:newsy,
        nlag=4, nhorz=17, addylag=false, firststagebyhorz=true, subset=df.wwii.==0,
        vce=HARVCE(EWC()))
    @test all(i->i==3, values(r1.estres.search.iopt))
    f1 = irf(r1, Cum(:y), Cum(:g))
    @test coef(f1)[1] ≈ 0.6969380791400279 atol=1e-8
    @test coef(f1)[9] ≈ 0.8444504165548141 atol=1e-8
    @test coef(f1)[17] ≈ 0.7343077490303411 atol=1e-8
    @test stderror(f1)[1] ≈ 0.4726399483395968 atol=1e-8
    @test stderror(f1)[9] ≈ 0.0858939264679245 atol=1e-8
    @test stderror(f1)[17] ≈ 0.1270379208490072 atol=1e-8

    r2 = lp(est, df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g),
        iv=Cum(:g)=>(:newsy, :g), nlag=4, nhorz=16, minhorz=1, addylag=false,
        firststagebyhorz=true, subset=df.wwii.==0, vce=HARVCE(EWC()))
    @test all(i->i==3, values(r2.estres.search.iopt))
    f2 = irf(r2, Cum(:y), Cum(:g))
    @test coef(f2)[1] ≈ -0.178375389687782 atol=1e-8
    @test coef(f2)[8] ≈ 0.2573602225267132 atol=1e-8
    @test coef(f2)[16] ≈ 0.23087128211782626 atol=1e-8
    @test stderror(f2)[1] ≈ 0.09138628413910047 atol=1e-8
    @test stderror(f2)[8] ≈ 0.03188098311909736 atol=1e-8
    @test stderror(f2)[16] ≈ 0.07079406030695458 atol=1e-8

    @test sprint(show, MIME("text/plain"), r2) == """
        LocalProjectionResult with 4 lags over 16 horizons:
        ──────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ──────────────────────────────────────────────────────────────────────────────
        Outcome variable:              Cum(y)    Minimum horizon:                    1
        Regressors:           Cum(g) constant    Lagged controls:            newsy y g
        Endogenous variable:           Cum(g)    Instruments:                  newsy g
        Kleibergen-Paap rk:   126.03 [<1e-54]    
        ──────────────────────────────────────────────────────────────────────────────
        Smooth Local Projection
        ──────────────────────────────────────────────────────────────────────────────
        Smoothing parameter:        100000.00    Smoothed regressor:            Cum(g)
        Polynomial order:                   3    Finite difference order:            3
        Selection criterion:            LOOCV    Selection algorithm:   DemmlerReinsch
        Leave-one-out CV:             1595.11    Generalized CV:               1594.28
        Akaike information:              7.37    Residual sum of squares:      1592.98
        ──────────────────────────────────────────────────────────────────────────────"""

    # Check the case with Cum but not IV
    est = SmoothLP(Cum(:g), 3, 3, search=grid([1e-8]))
    r3 = lp(est, df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g),
        nlag=4, nhorz=16, minhorz=1, addylag=false)
    # Compare results with the case without smoothing
    r3ns = lp(df, Cum(:y), xnames=Cum(:g), wnames=(:newsy, :y, :g),
        nlag=4, nhorz=16, minhorz=1, addylag=false)
    @test r3.B[1,1,:] ≈ r3ns.B[1,1,:] atol=1e-7

    # The case with Cum in variables that are not smoothed
    est = SmoothLP(:newsy, 3, 3, search=grid([1e-8]))
    r4 = lp(est, df, Cum(:y), xnames=(:newsy, Cum(:g)), wnames=(:newsy, :y, :g),
        nlag=4, nhorz=16, minhorz=1, addylag=false, subset=df.wwii.==0)
    r4ns = lp(df, Cum(:y), xnames=(:newsy, Cum(:g)), wnames=(:newsy, :y, :g),
        nlag=4, nhorz=16, minhorz=1, addylag=false, subset=df.wwii.==0)
    @test r4.B[1,1,:] ≈ r4ns.B[1,1,:] atol=1e-6

    @test_throws ArgumentError lp(est, df, (:y,:g), xnames=:newsy, wnames=:newsy)
    @test_throws ArgumentError lp(est, df, :y, xnames=:newsy, wnames=:newsy, vce=cluster(:iso))
end
