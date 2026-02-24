# Python References
#1: Water, EtOH, MeOH
# Composition 1 : [0.5 0.3 0.2]
# 	Logarithmic activity coefficients:
# 		O: 0.28
# 		CCO: 0.28
# 		CO: -0.03

# 	Excess Gibbs energy:
# 		g^E/RT = 0.22

# #2 DMSO, EtOH, Aspirin
# Composition 1 : [0.5 0.3 0.2]
# 	Logarithmic activity coefficients:
# 		OCS(=O)C: 0.11
# 		CCO: 0.03
# 		CC(=O)Oc1ccccc1C(=O)O: 0.59

# 	Excess Gibbs energy:
# 		g^E/RT = 0.18

# #3 Saccharin, Methanol, Chloroform
# Composition 1 : [0.5 0.3 0.2]
# 	Logarithmic activity coefficients:
# 		C1=CC=C2C(=C1)C(=O)NS2(=O)=O: 0.03
# 		CO: 0.41
# 		ClC(Cl)Cl: 0.33

# 	Excess Gibbs energy:
# 		g^E/RT = 0.20

@testset "multHANNA" begin
    # Systems to test, multHANNAs γ is raw, reference γ is ln(γ)
    systems = Dict(
        ["water", "ethanol", "methanol"]        => ([0.28, 0.28, -0.03], ["O", "CCO", "CO"]),
        ["dmso", "ethanol", "aspirin"]          => ([0.11, 0.03, 0.59], ["OCS(=O)C", "CCO", "CC(=O)Oc1ccccc1C(=O)O"]),
        ["saccharin", "methanol", "chloroform"] => ([0.03, 0.41, 0.33], ["C1=CC=C2C(=C1)C(=O)NS2(=O)=O", "CO", "ClC(Cl)Cl"])
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