module RDKitMinimalLibExt

using MLPROP
using RDKitMinimalLib: RDKitMinimalLib as RDK

function __init__()
end

# Canonicalize smiles
#TODO use Refs for _canonicalize (like for `MLPROP._GRAPPA[]`)
function _canonicalize(smiles)
    return RDK.get_smiles(RDK.get_mol(smiles))
end

function ChemBERTa.canonicalize(smiles::AbstractString; kwargs...)
    return _canonicalize(smiles)
end

# Get descriptors
function get_descriptors(smiles::AbstractString)
    mol = RDK.get_mol(smiles)
    return isnothing(mol) ? nothing : RDK.get_descriptors(mol)
end

end