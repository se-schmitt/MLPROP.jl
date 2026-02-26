using MLPROP, Clapeyron, PythonCall, ChemBERTa  

@testset "Models" begin
    include("test_hanna.jl")
    include("test_multHANNA.jl")

    @testset "GRAPPA" begin
        # Compare to https://ml-prop.mv.rptu.de
        model1 = GRAPPA("ethanol")
        @test model1.params.A[1] ≈ 15.197 rtol=1e-4
        @test model1.params.B[1] ≈ 2895.704 rtol=1e-5
        @test model1.params.C[1] ≈ -77.407 rtol=1e-5
        @test first(saturation_pressure(model1, 315.)) ≈ 20.282e3 rtol=1e-5
        crit1 = crit_pure(model1)
        @test crit1[1] == 513.92
        @test crit1[2] ≈ 5.237840522451958e6 rtol=1e-5

        model2 = GRAPPA(["toluene", "ome"]; userlocations=(; SMILES=["CC1=CC=CC=C1","COCOCOCOC"]))
        model2_1, model2_2 = split_model(model2)
        @test model2_1.params.A[1] ≈ 13.981 rtol=1e-4
        @test model2_1.params.B[1] ≈ 3060.594 rtol=1e-5
        @test model2_1.params.C[1] ≈ -55.588 rtol=1e-5
        @test first(saturation_pressure(model2_1, 315.)) ≈ 8.868e3 rtol=1e-4
        @test all(isnan.(crit_pure(model2_1)))
        @test model2_2.params.A[1] ≈ 14.795 rtol=1e-4
        @test model2_2.params.B[1] ≈ 3758.07 rtol=1e-5
        @test model2_2.params.C[1] ≈ -55.07 rtol=1e-5
        @test first(saturation_pressure(model2_2, 400.)) ≈ 49.404e3 rtol=1e-5
        @test all(isnan.(crit_pure(model2_2)))
    end
end