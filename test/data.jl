@testset "LPData" begin
    T = 100
    ys = Any[randn(T), randn(T)]
    xs = Any[randn(T), randn(T)]
    ws = ys
    fes0 = Any[]
    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 3, 0, nothing, nothing)
    Y, X, W, T1, eT = _makeYX(dt, 5)
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

    groups = [1:20, 21:T]
    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 3, 0, nothing, groups)
    @test dt.Xfull[:,1] == xs[1][vcat(4:20,24:T)]
    @test dt.Xfull[:,3] == ws[1][vcat(3:19,23:T-1)]
    Y, X, W, T1, eT = _makeYX(dt, 5)
    @test T1 == T-16
    @test size(Y) == (T1, 2)
    @test size(X) == (T1, 8)
    @test Y[:,1] == ys[1][vcat(9:20,29:T)]
    @test Y[:,2] == ys[2][vcat(9:20,29:T)]
    @test X[:,1] == xs[1][vcat(4:20-5,24:T-5)]
    @test X[:,3] == ws[1][vcat(3:20-6,23:T-6)]
    @test X[:,4] == ws[2][vcat(3:20-6,23:T-6)]
    @test X[:,8] == ws[2][vcat(1:20-8,21:T-8)]
    @test all(eT)

    sts = Any[rand(T), rand(T)]
    dt = LPData(ys, xs, ws, sts, fes0, nothing, 1, 0, nothing, nothing)
    Y, X, W, T1, eT = _makeYX(dt, 0)
    @test size(X) == (99, 6)
    @test X[:,1] == xs[1][2:end]
    @test X[:,2] == xs[2][2:end]
    @test X[:,3] == ys[1][1:99].*sts[1][2:end]
    @test X[:,4] == ys[1][1:99].*sts[2][2:end]
    @test X[:,5] == ys[2][1:99].*sts[1][2:end]
    @test X[:,6] == ys[2][1:99].*sts[2][2:end]
    @test all(eT)

    fes = Any[vcat(fill(1,20), fill(2,80))]
    dt = LPData(ys, xs, ws, nothing, fes, nothing, 3, 0, nothing, groups)
    Y, X, W, T1, eT = _makeYX(dt, 5)
    @test T1 == T-16
    @test size(Y) == (T1, 2)
    @test size(X) == (T1, 8)
    @test Y[1:12,1] ≈ ys[1][9:20] .- sum(ys[1][9:20])./12 atol=1e-8
    @test Y[13:T1,1] ≈ ys[1][29:T] .- sum(ys[1][29:T])./(T1-13+1) atol=1e-8
    @test Y[1:12,2] ≈ ys[2][9:20] .- sum(ys[2][9:20])./12 atol=1e-8
    @test Y[13:T1,2] ≈ ys[2][29:T] .- sum(ys[2][29:T])./(T1-13+1) atol=1e-8
    @test X[1:12,1] ≈ xs[1][4:15] .- sum(xs[1][4:15])./12 atol=1e-8
    @test X[13:T1,1] ≈ xs[1][24:T-5] .- sum(xs[1][24:T-5])./(T1-13+1) atol=1e-8
    @test X[1:12,8] ≈ ws[2][1:12] .- sum(ws[2][1:12])./12 atol=1e-8
    @test X[13:T1,8] ≈ ws[2][21:T-8] .- sum(ws[2][21:T-8])./(T1-13+1) atol=1e-8
    @test all(eT)

    pw = vcat(fill(4,20), fill(1,80))
    dt = LPData(ys, xs, ws, nothing, fes, pw, 3, 0, nothing, groups)
    Y, X, W, T1, eT = _makeYX(dt, 5)
    @test Y[1:12,1] ≈ 2.0.*(ys[1][9:20] .- sum(ys[1][9:20])./12) atol=1e-8
    @test Y[13:T1,1] ≈ ys[1][29:T] .- sum(ys[1][29:T])./(T1-13+1) atol=1e-8
    @test Y[1:12,2] ≈ 2.0.*(ys[2][9:20] .- sum(ys[2][9:20])./12) atol=1e-8
    @test Y[13:T1,2] ≈ ys[2][29:T] .- sum(ys[2][29:T])./(T1-13+1) atol=1e-8
    @test X[1:12,1] ≈ 2.0.*(xs[1][4:15] .- sum(xs[1][4:15])./12) atol=1e-8
    @test X[13:T1,1] ≈ xs[1][24:T-5] .- sum(xs[1][24:T-5])./(T1-13+1) atol=1e-8
    @test X[1:12,8] ≈ 2.0.*(ws[2][1:12] .- sum(ws[2][1:12])./12) atol=1e-8
    @test X[13:T1,8] ≈ ws[2][21:T-8] .- sum(ws[2][21:T-8])./(T1-13+1) atol=1e-8
    @test all(eT)

    dt = LPData(ys, xs, ws, sts, fes0, nothing, 1, 0, nothing, groups)
    Y, X, W, T1, eT = _makeYX(dt, 0)
    @test size(X) == (98, 6)
    @test X[:,1] == xs[1][vcat(2:20,22:T)]
    @test X[:,2] == xs[2][vcat(2:20,22:T)]
    @test X[:,3] == ys[1][vcat(1:19,21:T-1)].*sts[1][vcat(2:20,22:T)]
    @test X[:,4] == ys[1][vcat(1:19,21:T-1)].*sts[2][vcat(2:20,22:T)]
    @test X[:,5] == ys[2][vcat(1:19,21:T-1)].*sts[1][vcat(2:20,22:T)]
    @test X[:,6] == ys[2][vcat(1:19,21:T-1)].*sts[2][vcat(2:20,22:T)]
    @test all(eT)

    ys = Any[randn(T)]
    xs = Any[]
    ws = ys
    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, nothing, nothing)
    Y, X, W, T1, eT = _makeYX(dt, 0)
    @test T1 == 99
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 1)
    @test Y == reshape(ys[1][2:end], T1, 1)
    @test X == reshape(ws[1][1:end-1], T1, 1)
    @test all(eT)

    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, nothing, groups)
    Y, X, W, T1, eT = _makeYX(dt, 0)
    @test T1 == 98
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 1)
    @test Y == reshape(ys[1][vcat(2:20,22:T)], T1, 1)
    @test X == reshape(ws[1][vcat(1:19,21:T-1)], T1, 1)
    @test all(eT)

    ys[1][2], ys[1][3] = NaN, Inf
    xs = Any[convert(Vector{Union{Float64, Missing}}, randn(T))]
    xs[1][3] = missing
    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, nothing, nothing)
    Y, X, W, T1, eT = _makeYX(dt, 0)
    @test T1 == 96
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 2)
    @test Y == reshape(ys[1][5:end], T1, 1)
    @test X[:,1] == xs[1][5:end]
    @test X[:,2] == ws[1][4:end-1]
    @test eT == ((1:99).>3)

    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, nothing, groups)
    Y, X, W, T1, eT = _makeYX(dt, 0)
    @test T1 == 95
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 2)
    @test Y == reshape(ys[1][vcat(5:20,22:T)], T1, 1)
    @test X[:,1] == xs[1][vcat(5:20,22:T)]
    @test X[:,2] == ws[1][vcat(4:19,21:T-1)]
    @test eT == ((1:98).>3)

    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 1, 1, (1:100).<=90, nothing)
    Y, X, W, T1, eT = _makeYX(dt, 1)
    @test T1 == 85
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 2)
    @test Y == reshape(ys[1][6:90], T1, 1)
    @test X[:,1] == xs[1][5:89]
    @test X[:,2] == ws[1][4:88]
    # eT covers 2:99 in full data
    @test eT == ((1:98).>3) .& ((1:98).<=88)

    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 1, 1, (1:100).<=90, groups)
    Y, X, W, T1, eT = _makeYX(dt, 1)
    @test T1 == 83
    @test size(Y) == (T1, 1)
    @test size(X) == (T1, 2)
    @test Y == reshape(ys[1][vcat(6:20,23:90)], T1, 1)
    @test X[:,1] == xs[1][vcat(5:19,22:89)]
    @test X[:,2] == ws[1][vcat(4:18,21:88)]
    @test eT == ((1:96).>3) .& ((1:96).<=86)

    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, ((1:100).<60).|((1:100).>=70), nothing)
    Y, X, W, T1, eT = _makeYX(dt, 1)
    @test T1 == 83
    @test Y[:] == ys[1][vcat(6:59,72:T)]
    @test X[:,1] == xs[1][vcat(5:58,71:T-1)]
    @test X[:,2] == ws[1][vcat(4:57,70:T-2)]
    @test eT == ((1:98).>3) .& (((1:98).<58) .| ((1:98).>=70))

    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, ((1:100).<60).|((1:100).>=70), groups)
    Y, X, W, T1, eT = _makeYX(dt, 1)
    @test T1 == 81
    @test Y[:] == ys[1][vcat(6:20,23:59,72:T)]
    @test X[:,1] == xs[1][vcat(5:19,22:58,71:T-1)]
    @test X[:,2] == ws[1][vcat(4:18,21:57,70:T-2)]
    @test eT == ((1:96).>3) .& (((1:96).<56) .| ((1:96).>=68))

    dt = LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, ((1:100).<16).|((1:100).>=26), groups)
    Y, X, W, T1, eT = _makeYX(dt, 1)
    @test T1 == 83
    @test Y[:] == ys[1][vcat(6:15,28:T)]
    @test X[:,1] == xs[1][vcat(5:14,27:T-1)]
    @test X[:,2] == ws[1][vcat(4:13,26:T-2)]
    @test eT == ((1:96).>3) .& (((1:96).<14) .| ((1:96).>=24))

    dt = LPData(ys, xs, ws, nothing, fes, pw, 1, 0, ((1:100).<16).|((1:100).>=26), groups)
    Y, X, W, T1, eT = _makeYX(dt, 1)
    @test T1 == 83
    @test Y[1:10,1] ≈ 2.0.*(ys[1][6:15] .- sum(ys[1][6:15])./10) atol=1e-8
    @test Y[11:T1,1] ≈ ys[1][28:T] .- sum(ys[1][28:T])./(T1-10) atol=1e-8
    @test X[1:10,1] ≈ 2.0.*(xs[1][5:14] .- sum(xs[1][5:14])./10) atol=1e-8
    @test X[11:T1,1] ≈ xs[1][27:T-1] .- sum(xs[1][27:T-1])./(T1-10) atol=1e-8
    @test X[1:10,2] ≈ 2.0.*(ws[1][4:13] .- sum(ws[1][4:13])./10) atol=1e-8
    @test X[11:T1,2] ≈ ws[1][26:T-2] .- sum(ws[1][26:T-2])./(T1-10) atol=1e-8

    @test_throws ArgumentError LPData(ys, xs, ws, nothing, fes0, nothing, 0, 0, nothing, nothing)
    ys = Any[]
    @test_throws ArgumentError LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, nothing, nothing)
    ws = Any[]
    @test_throws ArgumentError LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, nothing, nothing)

    ys = Any[[Inf, NaN, 1.0]]
    xs = Any[]
    ws = ys
    @test_throws ArgumentError LPData(ys, xs, ws, nothing, fes0, nothing, 1, 0, nothing, nothing)

    @test sprint(show, dt) == "Data for local projection"
end
