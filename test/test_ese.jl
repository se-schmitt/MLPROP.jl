@testset "ESE" begin 
    p_iso = 1e5 

    # Compare to https://ml-prop.mv.rptu.de
    model_A = ESE(
        ["methylal", "dodecane"]; 
        userlocations=(;SMILES=["COCOC","CCCCCCCCCCCC"]),
        vismodel=[nothing,ConstantModel(Viscosity(), 0.0013153)]
    )    
    D_A = inf_diffusion_coefficient(model_A, p_iso, 300.; solute=1, solvent=2)*1e9
    @test D_A ≈ 1.68 rtol=1e-2
    
    # Test with EntropyScaling models
    model_B = ESE(["carbon dioxide", "water"])
    Tx = [299.16, 345.16]
    D_B_ref = [2.1565516995619975, 5.521765504779055]
    D_B = inf_diffusion_coefficient.(model_B, p_iso, Tx; solute=1, solvent=2)*1e9
    @test all(isapprox.(D_B, D_B_ref; rtol=1e-5))
end