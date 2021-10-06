@testset "kron_fast" begin
    p, q, n, m = 3, 4, 5, 6
    A = randn(p, q)
    B = randn(q*n, m)
    # Do not use I(n) to allow compatibility with Julia v1.0
    eye = Diagonal(ones(n))
    @test kron(eye, A) * B ≈ kron_fastl(A, B)
    @test kron(A, eye) * B ≈ kron_fastr(A, B)
end

@testset "getscore" begin
    T, K, N = 5, 3, 2
    X = randn(T, K)
    resid = randn(T, N)
    @test kron(X, ones(1, N)).*repeat(resid, outer=(1,K)) == getscore(X, resid)
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

    pa1 = copy(pa)
    pa1[2] = NaN
    c1, t1, e1 = hamilton_filter(pa1, :q)
    @test isfinite.(c1) == e1
    @test isfinite.(t1) == e1
    @test all(~e1[i] for i in 1:13)
    @test sum(e1) == length(c) - 13

    @test_throws ArgumentError hamilton_filter(pa, 0, 1)
    @test_throws ArgumentError hamilton_filter(pa, 1, 0)
    @test_throws ArgumentError hamilton_filter(pa[1:24], 8, 4)
    @test_throws ArgumentError hamilton_filter(pa, :n)
end
