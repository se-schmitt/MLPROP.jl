using ChemBERTa, DelimitedFiles, RDKitMinimalLib

@testset "ChemBERTa" begin
    # SMILES to test
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "C1=CC=C(C=C1)C2=C(C3=C(C(=NNC3=O)[O-])C(=N2)Cl)N.[Na+]",
        "CC[Hg]N1C(=O)C2C(C1=O)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl",
    ]

    # Loading the ChemBERTa model
    bert = ChemBERTa.load()

    # Calculating the embedding for a given SMILES and compare to Python reference
    for smiles in smiles_list
        embedding = bert(smiles)
        embedding_ref = readdlm("data/$(smiles).csv", Float32)[:]
        @test all((≈).(embedding, embedding_ref; atol=1f-5))
    end
end