using MLPROP, Clapeyron, PythonCall, ChemBERTa, JLD2, EntropyScaling

@testset "Models" begin
    include("test_hanna.jl")

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

    #TODO revise tests
    @testset "ESE" begin
        p_iso = 1e5
        
        validation_data=Dict(
            [
                ("299.16",[1.6482543537961485;0.931919139410536;2.2720926704344366;2.1565516995619975]),
                ("319.16",[2.398242890812778;1.499678069198921;3.4423174568853114;3.4051672004992155]),
                ("330.16",[2.8835049909007537;1.883458270815633;4.266043969220459;4.233824652227727]),
                ("335.16",[3.1220121860028516;2.0752810400719994;4.689285161702368;4.643207359176968]),
                ("345.16",[3.633999694982114;2.4922799695456694;5.636959093505703;5.521765504779055])
            ]
        )

        Tx = [299.16;319.16;330.16;335.16;345.16]
        
        η_model_dodecane = RefpropRESModel("dodecane")
        η_fun_dodecane = T -> viscosity(η_model_dodecane, 1e5, T)

        η_model_hexadecane= RefpropRESModel("hexadecane")
        η_fun_hexadecane = T -> viscosity(η_model_hexadecane, 1e5, T)

        η_model_ethanol = RefpropRESModel("ethanol")
        η_fun_ethanol = T -> viscosity(η_model_ethanol, 1e5, T)

        η_model_water = RefpropRESModel("water")
        η_fun_water = T -> viscosity(η_model_water, 1e5, T)

        model_diolane_hexadecane=ESE("C1OCOC1","CCCCCCCCCCCCCCCC",η_fun_hexadecane)
        model_Acetonitrile_ethanol=ESE("CC#N","CCO",η_fun_ethanol)
        model_Carbondioxide_water=ESE("O=C=O","O",η_fun_water)
        model_methylal_dodecane=ESE("COCOC","CCCCCCCCCCCC",η_fun_dodecane)


        Diff_methylal_dodecane=Diffusion.(model_methylal_dodecane,p_iso,Tx)*10^9
        Diff_diolane_hexadecane=Diffusion.(model_diolane_hexadecane,p_iso,Tx)*10^9
        Diff_Acetonitrile_ethaol=Diffusion.(model_Acetonitrile_ethanol,p_iso,Tx)*10^9
        Diff_Carbondioxide_water=Diffusion.(model_Carbondioxide_water,p_iso,Tx)*10^9


        @testset "MLPROP.jl" begin
        @test Diff_methylal_dodecane[1] ≈ get(validation_data,"299.16",0)[1] atol=1e-5
        @test Diff_diolane_hexadecane[1] ≈ get(validation_data,"299.16",0)[2] atol=1e-5
        @test Diff_Acetonitrile_ethaol[1] ≈ get(validation_data,"299.16",0)[3] atol=1e-5
        @test Diff_Carbondioxide_water[1] ≈ get(validation_data,"299.16",0)[4] atol=1e-5

        @test Diff_methylal_dodecane[2] ≈ get(validation_data,"319.16",0)[1] atol=1e-5
        @test Diff_diolane_hexadecane[2] ≈ get(validation_data,"319.16",0)[2] atol=1e-5
        @test Diff_Acetonitrile_ethaol[2] ≈ get(validation_data,"319.16",0)[3] atol=1e-5
        @test Diff_Carbondioxide_water[2] ≈ get(validation_data,"319.16",0)[4] atol=1e-5

        @test Diff_methylal_dodecane[3] ≈ get(validation_data,"330.16",0)[1] atol=1e-5
        @test Diff_diolane_hexadecane[3] ≈ get(validation_data,"330.16",0)[2] atol=1e-5
        @test Diff_Acetonitrile_ethaol[3] ≈ get(validation_data,"330.16",0)[3] atol=1e-5
        @test Diff_Carbondioxide_water[3] ≈ get(validation_data,"330.16",0)[4] atol=1e-5

        @test Diff_methylal_dodecane[4] ≈ get(validation_data,"335.16",0)[1] atol=1e-5
        @test Diff_diolane_hexadecane[4] ≈ get(validation_data,"335.16",0)[2] atol=1e-5
        @test Diff_Acetonitrile_ethaol[4] ≈ get(validation_data,"335.16",0)[3] atol=1e-5
        @test Diff_Carbondioxide_water[4] ≈ get(validation_data,"335.16",0)[4] atol=1e-5

        @test Diff_methylal_dodecane[5] ≈ get(validation_data,"345.16",0)[1] atol=1e-5
        @test Diff_diolane_hexadecane[5] ≈ get(validation_data,"345.16",0)[2] atol=1e-5
        @test Diff_Acetonitrile_ethaol[5] ≈ get(validation_data,"345.16",0)[3] atol=1e-5
        @test Diff_Carbondioxide_water[5] ≈ get(validation_data,"345.16",0)[4] atol=1e-5
        end
    end
end