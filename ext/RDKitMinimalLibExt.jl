module RDKitMinimalLibExt

using MLPROP
using RDKitMinimalLib: RDKitMinimalLib as RDK

function __init__()
    MLPROP._get_descriptors[] = _get_descriptors_rdk
end

# Get descriptors
function _get_descriptors_rdk(smiles::AbstractString)
    mol = RDK.get_mol(smiles)
    return isnothing(mol) ? nothing : RDK.get_descriptors(mol)
end

end