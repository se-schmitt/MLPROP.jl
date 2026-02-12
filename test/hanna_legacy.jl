using MLPROP, ChemBERTa, Clapeyron
# Test components
# Water (1) and Ethanol (2)
# (1): "O"
# (2): "CCO"
# Gamma First Comp.:  1.455121
# Gamma Second Comp: 1.248844

# DMSO (1) and Water (2)
# (1): "CS(=O)C"
# (2): "O"
# Gamma First Comp.:  0.756031
# Gamma Second Comp: 0.552130

# Aspirin (1) and Methanol (2)
# (1): "CC(=O)Oc1ccccc1C(=O)O"
# (2): "CO"
# Gamma First Comp.:  1.197029
# Gamma Second Comp: 1.551961

# Saccharin (1) and Methanol (2)
# (1): "C1=CC=C2C(=C1)C(=O)NS2(=O)=O"
# (2): "CO"
# Gamma First Comp.:  1.185503
# Gamma Second Comp: 1.335264


@testset "HANNA_legacy" begin
    # System to test
    systems = Dict(
        ["water", "ethanol"]     => [1.455121, 1.248844],                                
        ["DMSO","water"]         => [0.756031, 0.552130],                          
        ["aspirin","methanol"]   => [1.197029, 1.551961],              
        ["saccharin","methanol"] => [1.185503, 1.335264],      
    )

    # Calculating the gammas for a given SMILES-pair and compare to Python reference
    for (system, γs_ref) in systems
        model = MLPROP.HANNA(system)
        γs = activity_coefficient(model, 1e5, 300., [.5,.5])
        @test γs[1] ≈ γs_ref[1] rtol=1e-5
        @test γs[2] ≈ γs_ref[2] rtol=1e-5
    end
end