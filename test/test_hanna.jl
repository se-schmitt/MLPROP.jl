@testset "ogHANNA" begin
    # System to test
    systems = Dict(
        ["water", "ethanol"]      => ([1.455121, 1.248844], ["O", "CCO"]),
        ["DMSO", "water"]         => ([0.756031, 0.552130], ["CS(=O)C", "O"]),
        ["aspirin", "methanol"]   => ([1.197029, 1.551961], ["CC(=O)Oc1ccccc1C(=O)O", "CO"]),
        ["saccharin", "methanol"] => ([1.185503, 1.335264], ["C1=CC=C2C(=C1)C(=O)NS2(=O)=O", "CO"]),
    )

    # Calculating the gammas for a given SMILES-pair and compare to Python reference
    for (system_i, (γs_ref_i, smiles_i)) in systems
        # Use Clapeyron.jl database
        model = ogHANNA(system_i)
        γs_i = activity_coefficient(model, 1e5, 300., [.5,.5])
        @test γs_i[1] ≈ γs_ref_i[1] rtol=1e-5
        @test γs_i[2] ≈ γs_ref_i[2] rtol=1e-5

        # Use `userlocations` keyword
        model_smiles = ogHANNA(["comp A", "comp B"]; userlocations=(;SMILES=smiles_i))
        γs_i_smiles = activity_coefficient(model_smiles, 1e5, 300., [.5,.5])
        @test γs_i_smiles[1] ≈ γs_ref_i[1] rtol=1e-5
        @test γs_i_smiles[2] ≈ γs_ref_i[2] rtol=1e-5
    end
end

@testset "multHANNA" begin
    # Systems to test, multHANNAs γ is raw, reference γ is ln(γ)
    systems = Dict(
        ["water", "ethanol", "methanol"]        => ([0.277717203, 0.278312653, -0.033341952], ["O", "CCO", "CO"]),
        ["dmso", "ethanol", "aspirin"]          => ([-0.047125854, -0.071092166, 0.233126670], ["CS(=O)C", "CCO", "CC(=O)Oc1ccccc1C(=O)O"]),
        ["saccharin", "methanol", "chloroform"] => ([0.028507818, 0.408137769, 0.330980510], ["C1=CC=C2C(=C1)C(=O)NS2(=O)=O", "CO", "ClC(Cl)Cl"])
    )

    # Calculating the gammas for a given SMILES-pair and compare to Python reference
    for (system_i, (lnγs_ref_i, smiles_i)) in systems
        # Use Clapeyron.jl database
        model = multHANNA(system_i)
        γs_i = activity_coefficient(model, 1e5, 300., [.5, .3, .2])
        @test γs_i[1] ≈ exp(lnγs_ref_i[1]) rtol=1e-5
        @test γs_i[2] ≈ exp(lnγs_ref_i[2]) rtol=1e-5
        @test γs_i[3] ≈ exp(lnγs_ref_i[3]) rtol=1e-5

        # Use `userlocations` keyword
        model_smiles = multHANNA(["comp A", "comp B", "comp C"]; userlocations=(;SMILES=smiles_i))
        γs_i_smiles = activity_coefficient(model_smiles, 1e5, 300., [.5, .3, .2])
        @test γs_i_smiles[1] ≈ exp(lnγs_ref_i[1]) rtol=1e-5
        @test γs_i_smiles[2] ≈ exp(lnγs_ref_i[2]) rtol=1e-5
        @test γs_i_smiles[3] ≈ exp(lnγs_ref_i[3]) rtol=1e-5
    end
end