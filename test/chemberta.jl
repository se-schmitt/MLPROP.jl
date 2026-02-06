using ChemBERTa, DelimitedFiles

@testset "ChemBERTa" begin
    # SMILES to test
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "C1=CC=C(C=C1)C2=C(C3=C(C(=NNC3=O)[O-])C(=N2)Cl)N.[Na+]",
        "CC[Hg]N1C(=O)C2C(C1=O)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl",
    ]
    canonical_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O"
        "Nc1c(-c2ccccc2)nc(Cl)c2c([O-])n[nH]c(=O)c12.[Na+]"
        "CC[Hg]N1C(=O)C2C(C1=O)C1(Cl)C(Cl)=C(Cl)C2(Cl)C1(Cl)Cl"
    ]

    # Test canonization
    if Sys.islinux()
        @testset "RDKitMinimalLibExt" begin
            using RDKitMinimalLib
            for i in eachindex(smiles_list)
                @test ChemBERTa.canonicalize.(smiles_list[i]) == canonical_smiles[i]
            end
        end
    end

    @testset "PythonCall" begin
        using PythonCall
        for i in eachindex(smiles_list)
            @test ChemBERTa.canonicalize.(smiles_list[i]) == canonical_smiles[i]
        end
    end

    # Loading the ChemBERTa model
    bert = ChemBERTa.load()

    # Calculating the embedding for a given SMILES and compare to Python reference
    for smiles in smiles_list
        embedding = bert(smiles)
        embedding_ref = readdlm("data/$(smiles).csv", Float32)[:]
        @test all((≈).(embedding, embedding_ref; atol=1f-5))
    end
end