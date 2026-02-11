module RDKitMinimalLibExt

using ChemBERTa
using RDKitMinimalLib: RDKitMinimalLib as RDK

function __init__()
    ChemBERTa._canonicalize[] = _canonicalize_rdk
end

function _canonicalize_rdk(smiles)
    return RDK.get_smiles(RDK.get_mol(smiles))
end

end