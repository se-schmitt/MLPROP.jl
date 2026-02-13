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