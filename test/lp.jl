@testset "OLS" begin
    T, N, K = 100, 2, 4
    X = randn(T, K)
    b = randn(K, N)
    Y = X * b
    m = ols(Y, X)
    @test m.coef ≈ b
    @test modelmatrix(m) === X
    @test coef(m) === m.coef
    @test residuals(m) === m.resid
end

@testset "reg" begin
    T, N, K = 100, 2, 4
    X = randn(T, K)
    b = randn(K, N)
    Y = X * b + randn(T, N)
    (b, v) = reg(Y, X, SimpleVCE())
    @test size(b) == (K, N)
    @test size(v) == (K*N, K*N)
end

@testset "LocalProjectionResult" begin
    B = [randn(5, 2), randn(5, 2)]
    V = [randn(10, 10), randn(10, 10)]
    r = LocalProjectionResult(B, V, [1], HRVCE(), [:y1, :y2], [:x], [:w1, :w2], Dict(:y1=>1, :y2=>2), Dict(:x=>1, :w1=>2, :w2=>3), 2, true)
    @test coef(r, 1, :y1, :x) == B[1][1,1]
    @test coef(r, 2, :y2, :w1) == B[2][2,2]
    @test coef(r, 2, :y1, :w1, 2) == B[2][4,1]
    @test vcov(r, 1, :y1, :x) == V[1][1,1]
    @test vcov(r, 2, :y2, :x, 1, :y1, :w1) == V[2][2,3]
    @test vcov(r, 2, :y2, :w1, 2, :y1, :w1, 1) == V[2][8,3]
    @test vcov(r, 2, :y2, :w2, 2, :y1, :w1) == V[2][10,7]

    @test sprint(show, r) == "LocalProjectionResult"
    @test sprint(show, MIME("text/plain"), r) == """
        LocalProjectionResult with 2 lags over 2 horizons:
          outcome names: y1 y2
          regressor name: x
          lagged control names: w1 w2"""
    r = LocalProjectionResult([ones(1,1)], [ones(1,1)], [1], HRVCE(), [:y], Symbol[], [:w], Dict(:y=>1), Dict(:w=>2), 1, true)
    @test sprint(show, MIME("text/plain"), r) == """
        LocalProjectionResult with 1 lag over 1 horizon:
          outcome name: y
          lagged control name: w"""
end

@testset "_makeYX" begin
    T, L, H = 100, 3, 5
    ys = [randn(T), randn(T)]
    xs = [randn(T), randn(T)]
    ws = ys
    Y, X, T1 = _makeYX(ys, xs, ws, L, H)
    @test T1 == T-L-H
    @test size(Y) == (T1, 2)
    @test size(X) == (T1, 8)
    @test Y[:,1] == ys[1][9:end]
    @test Y[:,2] == ys[2][9:end]
    @test X[:,1] == xs[1][4:end-5]
    @test X[:,3] == ws[1][3:end-6]
    @test X[:,4] == ws[2][3:end-6]
    @test X[:,8] == ws[2][1:end-8]

    ys = (randn(T),)
    xs = ()
    ws = ys
    Y, X, T1 = _makeYX(ys, xs, ws, 1, 0)
    @test T1 == 99
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 1)
    @test Y == reshape(ys[1][2:end], T1, 1)
    @test X == reshape(ws[1][1:end-1], T1, 1)

    ys[1][2], ys[1][3] = NaN, Inf
    xs = (convert(Vector{Union{Float64, Missing}}, randn(T)),)
    xs[1][3] = missing
    Y, X, T1 = _makeYX(ys, xs, ws, 1, 0)
    @test T1 == 96
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 2)
    @test Y == reshape(ys[1][5:end], T1, 1)
    @test X[:,1] == xs[1][5:end]
    @test X[:,2] == ws[1][4:end-1]

    ys = ()
    @test_throws ArgumentError _makeYX(ys, xs, ws, 1, 0)
    ys = (randn(T-1),)
    @test_throws ArgumentError _makeYX(ys, xs, ws, 1, 0)
    ws = ()
    @test_throws ArgumentError _makeYX(ys, xs, ws, 1, 0)
    @test_throws ArgumentError _makeYX(ys, xs, ws, 0, 0)

    ys = ([Inf, NaN, 1.0],)
    xs = ()
    ws = ys
    @test_throws ArgumentError _makeYX(ys, xs, ws, 1, 0)
end

@testset "lp" begin
    T, L, H = 100, 3, 5
    ys = [randn(T), randn(T)]
    xs = [randn(T), randn(T)]
    ws = ys
    b, v = _lp(ys, xs, ws, 3, 5, SimpleVCE())
    @test size(b) == (8, 2)
    @test size(v) == (16, 16)

    # Reproduce estimates from example based on Gertler & Karadi (2015)
    # Specification is non-augmented local projection
    # EHW standard errors
    # https://github.com/jm4474/Lag-augmented_LocalProjections/blob/master/examples/gk.m
    # The intercept estimated will be the same if not detrend the level before estimation
    df = exampledata(:gk)
    ns = (:logcpi, :logip, :ff, :ebp, :ff4_tc)
    r = lp(df, :ebp, wnames=ns, nlag=12, nhorz=48)
    @test coef(r, 1, :ebp, :ff4_tc) ≈ 0.394494467439933
    @test coef(r, 5, :ebp, :ff4_tc) ≈ 0.147288229090455
    @test sqrt(vcov(r, 1, :ebp, :ff4_tc)) ≈ 0.473834526724786
    @test sqrt(vcov(r, 5, :ebp, :ff4_tc)) ≈ 0.737092370022356
    # Compare additional estimates obtained from Matlab lp before being turned into irf
    # Run in gk.m: lp(Y, 11, 1, 4, false, false)
    @test coef(r, 1, :ebp, :constant) ≈ -2.945365821418727
    @test coef(r, 1, :ebp, :logip) ≈ -0.060094199554036
    @test vcov(r, 1, :ebp, :constant) ≈ 0.641933721888132
    @test vcov(r, 1, :ebp, :logcpi) ≈ 0.007331238288153
    @test vcov(r, 1, :ebp, :logcpi, 12) ≈ 0.004780116443880

    f = irf(r, :ebp, :ff4_tc)
    @test coef(f)[1] ≈ 0.394494467439933
    @test coef(f)[20] ≈ -0.552849402428304
    @test stderror(f)[1] ≈ 0.473834526724786
    @test stderror(f)[20] ≈ 1.005046789996172
    ci = confint(f)
    @test ci[1][1] ≈ -0.384893972486403
    @test ci[2][1] ≈ 1.173882907366730
    @test ci[1][20] ≈ -2.206004261810359
    @test ci[2][20] ≈ 1.100305456953579

    ns1 = [3, :logip, 5, :ff4_tc]
    r1 = lp(df, 11, wnames=ns1, nlag=12, nhorz=1)
    @test r1.B[1][1:3] ≈ r.B[1][1:3]
    @test r1.V[1][1:3,1:3] ≈ r.V[1][1:3,1:3]
end
