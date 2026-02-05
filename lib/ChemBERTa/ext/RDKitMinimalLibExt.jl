module RDKitMinimalLibExt

using ChemBERTa
using RDKitMinimalLib: RDKitMinimalLib as RDK

function __init__()
    if Base.get_extension(Main.ChemBERTa, :PythonCallExt) isa Module
        @warn """
        Function `ChemBERTa.canonicalize` defined by `RDKitMinimalLibExt` and `PythonCallExt`!
        `RDKitMinimalLibExt` is used.
        """
    end
end

function _canonicalize(smiles)
    return RDK.get_smiles(RDK.get_mol(smiles))
end

function ChemBERTa.canonicalize(smiles::AbstractString; kwargs...)
    return _canonicalize(smiles)
end

end