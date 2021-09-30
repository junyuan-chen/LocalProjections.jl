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
