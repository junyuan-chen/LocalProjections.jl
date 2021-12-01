@testset "kron_fast" begin
    p, q, n, m = 3, 4, 5, 6
    A = randn(p, q)
    B = randn(q*n, m)
    # Do not use I(n) to allow compatibility with Julia v1.0
    eye = Matrix{Float64}(I, n, n)
    @test kron(eye, A) * B ≈ kron_fastl(A, B)
    @test kron(A, eye) * B ≈ kron_fastr(A, B)
end

@testset "getscore" begin
    T, K, N = 5, 3, 2
    X = randn(T, K)
    resid = randn(T, N)
    @test getscore(X, resid) == kron(X, ones(1, N)).*repeat(resid, outer=(1,K))
    resid = randn(T)
    @test getscore(X, resid) == X.*resid
end

@testset "hamilton_filter" begin
    df = exampledata(:hp)
    pa = 100.0.*log.(df.payrolls_a)
    c, t, e = hamilton_filter(pa, 8, 4)
    @test length(c) == length(pa)
    @test all(isnan(c[i]) for i in 1:11)
    # Compare results with Matlab code from replication files
    # Need to replace `b = inv(X'*X)*X'*y(p+h:T,1)` in `regcyc.m` with `b = X\y(p+h:T,1)`
    @test c[12] ≈ -7.828544973785029
    @test c[200] ≈ 1.462770379340782
    @test all(~e[i] for i in 1:11)
    @test sum(e) == length(c) - 11

    c1, t1 = hamilton_filter(pa, :q)
    @test isequal(c1, c)
    @test isequal(t1, t)
    c1, t1 = hamilton_filter(pa, 24, 12)
    c2, t2 = hamilton_filter(pa, :m)
    @test isequal(c2, c1)
    @test isequal(t2, t1)
    c1, t1 = hamilton_filter(pa, 2, 1)
    c2, t2 = hamilton_filter(pa, :y)
    @test isequal(c2, c1)
    @test isequal(t2, t1)

    T = length(pa)
    pa1 = copy(pa)
    pa1[2] = NaN
    c1, t1, e1 = hamilton_filter(pa1, :q)
    @test isfinite.(c1) == e1
    @test isfinite.(t1) == e1
    @test e1 == ((1:T).>13)

    c1, t1, e1 = hamilton_filter(pa1, :q, subset=1:T.<=100)
    @test isfinite.(c1) == e1
    @test isfinite.(t1) == e1
    @test e1 == ((1:T).>13).&((1:T).<=100)

    @test_throws ArgumentError hamilton_filter(pa, 0, 1)
    @test_throws ArgumentError hamilton_filter(pa, 1, 0)
    @test_throws ArgumentError hamilton_filter(pa[1:24], 8, 4)
    @test_throws ArgumentError hamilton_filter(pa, :n)
end

@testset "Cum" begin
    df = exampledata(:gk)
    T = size(df, 1)
    c = Cum(:ff4_tc)
    @test _geto(c) == :ff4_tc
    v = getcolumn(df, c)
    @test v === Cum(df.ff4_tc, nothing)
    @test size(v, 1) == size(df.ff4_tc, 1)
    @test length(v) == length(df.ff4_tc)
    @test isequal(vec(Cum(df.ff4_tc), nothing, :x, 0, Float64), coalesce.(df.ff4_tc, NaN))

    r = copy(df.ff4_tc)
    r[1:end-2] += df.ff4_tc[2:end-1] + df.ff4_tc[3:end]
    r[end-1:end] .= NaN
    v = vec(Cum(df.ff4_tc), nothing, :x, 2, Float64)
    @test v[.~isnan.(v)] ≈ r[.~isnan.(v)]
    v = vec(Cum(df.ff4_tc), (1:T.<=200).|(1:T.>220), :x, 2, Float64)
    @test v[.~isnan.(v)] ≈ r[.~isnan.(v)]
    @test isnan.(v[191:230]) == ((1:40).>8).&((1:40).<=30)
    r = copy(df.ff4_tc)
    r[3:end] += df.ff4_tc[1:end-2] + df.ff4_tc[2:end-1]
    r[1:2] .= NaN
    v = vec(Cum(df.ff4_tc), nothing, :y, 2, Float64)
    @test v[.~isnan.(v)] ≈ r[.~isnan.(v)]
    ss = (1:T.<=200).|(1:T.>220)
    v = vec(Cum(df.ff4_tc), ss, :y, 2, Float64)
    @test v[.~isnan.(v)] ≈ r[.~isnan.(v)]
    @test isnan.(v[191:230]) == ((1:40).>10).&((1:40).<=32)

    df.s = df.ff4_tc.>0
    c1 = Cum(:ff, :s)
    @test _geto(c1) == :ff
    v1 = getcolumn(df, c1)
    @test v1 === Cum(df.ff, df.s)
    @test size(v1, 1) == size(df.ff, 1)
    @test length(v1) == length(df.ff)

    @test isequal(vec(v1, nothing, :x, 0, Float64), coalesce.(df.ff.*df.s, NaN))
    @test isequal(vec(v1, nothing, :y, 0, Float64), coalesce.(df.ff, NaN))
    @test isequal(view(vec(v1, ss, :x, 0, Float64), ss), coalesce.(view(df.ff.*df.s, ss), NaN))
    @test isequal(view(vec(v1, ss, :y, 0, Float64), ss), coalesce.(view(df.ff, ss), NaN))

    @test _toint(df, c) === Cum(16)
    @test _toint(df, c1) === Cum(5, 20)
    @test _toname(df, Cum(16)) === c
    @test _toname(df, Cum(5, 20)) === c1
    @test sprint(show, c) == "Cum(ff4_tc)"
    @test sprint(show, c1) == "Cum(ff, s)"
end
