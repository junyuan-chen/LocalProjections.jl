@testset "Ridge" begin
    a2 = rand(2,2)
    m = Ridge(rand(5), rand(5,2), a2, rand(2), a2, 1.0, a2, nothing, 1.0,
        rand(2), rand(5), a2, 1.0, 1)
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
    ys, ss, xs = [randn(T)], [randn(T)], [randn(T)]
    ws = ys
    res, X, T1, e1, e2 = _makeYSr(ys, ss, xs, ws, 3, 5, nothing)
    @test T1 == T-8
    @test size(res) == (T1, 2)
    @test all(e1)
    @test all(e2)

    xs = ()
    res, X, T1, e1, e2 = _makeYSr(ys, ss, xs, ws, 1, 0, nothing)
    @test T1 == 99
    @test size(res) == (T1, 2)
    @test all(e1)
    @test all(e2)

    ys[1][2], ys[1][3] = NaN, Inf
    xs = (convert(Vector{Union{Float64, Missing}}, randn(T)),)
    xs[1][3] = missing
    res, X, T1, e1, e2 = _makeYSr(ys, ss, xs, ws, 1, 0, nothing)
    @test T1 == 96
    @test size(res) == (T1, 2)
    @test e1 == ((1:99).>3)
    @test all(e2)

    res, X, T1, e1, e2 = _makeYSr(ys, ss, xs, ws, 1, 1, ((1:100).<60).|((1:100).>=70))
    @test T1 == 83
    @test e1 == ((1:98).>3) .& (((1:98).<58) .| ((1:98).>=70))
    @test all(e2)

    ys, xs = [randn(T)], [randn(T)]
    ss[1][2], ss[1][4] = NaN, Inf
    ws = ys
    res, X, T1, e1, e2 = _makeYSr(ys, ss, xs, ws, 2, 0, nothing)
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
    @test ci1[1][1] ≈ -0.04758915684832821
    @test ci1[2][1] ≈ 0.8758551271581806
    @test ci1[1][10] ≈ -0.35800938178031794
    @test ci1[2][10] ≈ 2.265871824705655

    @test sprint(show, MIME("text/plain"), r1) == """
        LocalProjectionResult with 12 lags over 48 horizons:
        ────────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ────────────────────────────────────────────────────────────────────────────────
        Outcome variable:                  ebp    Minimum horizon:                     0
        Regressors:            ff4_tc constant    
        Lagged controls:                                      logcpi logip ff ebp ff4_tc
        ────────────────────────────────────────────────────────────────────────────────
        Smooth Local Projection
        ────────────────────────────────────────────────────────────────────────────────
        Smoothing parameter:              0.00    Smoothed regressor:             ff4_tc
        Polynomial order:                    3    Finite difference order:             3
        Selection criterion:               GCV    Selection algorithm:    DemmlerReinsch
        Leave-one-out CV:              2212.02    Generalized CV:                2212.25
        Akaike information:               7.70    Residual sum of squares:       2193.42
        ────────────────────────────────────────────────────────────────────────────────"""

    # Compare estimates with Matlab results from Barnichon and Brownlees (2019)
    df = exampledata(:bb)
    ns = (:ir, :pi, :yg)
    # 194 is their optimal smoothing parameter
    est = SmoothLP(:ir, 3, 2, search=grid(194), criterion=LOOCV())
    r1 = lp(est, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1)
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
    @test ci1[1][1] ≈ -0.8571544691033884
    @test ci1[2][1] ≈ 0.3850826992477342
    @test ci1[1][10] ≈ 0.020165100998330715
    @test ci1[2][10] ≈ 0.34552900750920656

    # Compare results based on DemmlerReinsch and DirectSolve
    gbb = 194.0.*(1:0.5:10)
    est1 = SmoothLP(:ir, 3, 2, search=grid(gbb), criterion=LOOCV())
    r1 = lp(est1, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1)
    @test r1.estres.λ == 194
    @test r1.estres.search.i == 1
    @test r1.estres.rss ≈ sum(r1.estres.m.resid.^2)
    est2 = SmoothLP(:ir, 3, 2, search=grid(gbb, algo=DirectSolve()), criterion=GCV())
    r2 = lp(est2, df, :yg, xnames=ns, wnames=ns, nlag=4, nhorz=20, minhorz=1)
    @test r2.estres.λ == 194
    @test r2.estres.search.i == 1
    @test r2.estres.rss ≈ sum(r2.estres.m.resid.^2)
    @test r1.estres.search.θs ≈ r2.estres.search.θs atol=1e-2
    @test r1.estres.search.loocv ≈ r2.estres.search.loocv atol=1e-1
    @test r1.estres.search.rss ≈ r2.estres.search.rss atol=1e-1
    @test r1.estres.search.gcv ≈ r2.estres.search.gcv atol=1e-1
    @test r1.estres.search.aic ≈ r2.estres.search.aic atol=1e-6
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

    # Compare estimates from LeastSquareLP
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
    @test ci4[1][1] ≈ -0.6790290214946426
    @test ci4[2][1] ≈ 0.20695725163898837
    @test ci4[1][10] ≈ -0.06529955673073387
    @test ci4[2][10] ≈ 0.4309936652382712

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
        ────────────────────────────────────────────────────────────────────────────────
        Variable Specifications
        ────────────────────────────────────────────────────────────────────────────────
        Outcome variable:                   yg    Minimum horizon:                     1
        Regressors:          ir pi yg constant    Lagged controls:              ir pi yg
        ────────────────────────────────────────────────────────────────────────────────
        Smooth Local Projection
        ────────────────────────────────────────────────────────────────────────────────
        Smoothing parameter:            194.00    Smoothed regressor:                 ir
        Polynomial order:                    3    Finite difference order:             2
        Selection criterion:             LOOCV    Selection algorithm:    DemmlerReinsch
        Leave-one-out CV:             34530.21    Generalized CV:               34511.09
        Akaike information:              10.45    Residual sum of squares:      34379.97
        ────────────────────────────────────────────────────────────────────────────────"""
end
