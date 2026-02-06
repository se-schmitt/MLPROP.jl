module PythonCallExt

using ChemBERTa
using PythonCall

const chem = Ref{Py}()
  
function __init__()
    if isdefined(Main, :ChemBERTa) && Base.get_extension(Main.ChemBERTa, :RDKitMinimalLibExt) isa Module
        @warn """
        Function `ChemBERTa.canonicalize` defined by `PythonCallExt` and `RDKitMinimalLibExt`!
        `PythonCallExt` is used.
        """
    end

    chem[] = pyimport("rdkit.Chem")
end

# RDKit functions
function _get_mol(smiles)
    return chem[].MolFromSmiles(smiles)
end

function _get_smiles(mol)
    return chem[].MolToSmiles(mol)
end

# Canonization
function _canonicalize(smiles)
    mol = _get_mol(smiles)
    string(mol) == "None" && error("Invalid SMILES: '$(smiles)'!")
    return string(_get_smiles(mol))
end

function ChemBERTa.canonicalize(smiles::AbstractString; kwargs...)
    return _canonicalize(smiles)
end

end