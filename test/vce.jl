@testset "SimpleVCE" begin
    @test sprint(show, MIME("text/plain"), SimpleVCE()) ==
        "Simple variance-covariance estimator"
end

@testset "HRVCE" begin
    @test criticalvalue(HRVCE(), 0.95, 100, 100) ≈ 1.9839715185235516
    @test pvalue(HRVCE(), 1.96, 100, 100) ≈ 0.05277890136622965
    @test sprint(show, MIME("text/plain"), HRVCE()) ==
        "Heteroskedasticity-robust variance-covariance estimator"
end

@testset "EWC" begin
    @test EWC() === EqualWeightedCosine()
    @test sprint(show, MIME("text/plain"), EWC()) == "Equal-weighted cosine transform"
    v = HARVCE(EWC())
    @test criticalvalue(v, 0.95, 100, 10) ≈ 2.262157162798205
    @test pvalue(v, 1.96, 100, 10) ≈ 0.08164440546041651
end

@testset "HARVCE" begin
    v = HARVCE(EWC())
    @test sprint(show, MIME("text/plain"), v) == """
        Heteroskedasticity-autocorrelation-robust variance-covariance estimator:
          Long-run variance estimator: Equal-weighted cosine transform"""
end
