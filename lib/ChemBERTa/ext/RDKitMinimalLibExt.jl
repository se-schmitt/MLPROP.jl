module RDKitMinimalLibExt

using RDKitMinimalLib: RDKitMinimalLib as RDK
using ChemBERTa: ChemBERTa as CB

function CB.canonicalize(smiles)
    return RDK.get_smiles(RDK.get_mol(smiles))
end

end