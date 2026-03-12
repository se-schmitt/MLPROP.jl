module PythonCallExt

using MLPROP
using PythonCall
const CL = MLPROP.Clapeyron

const chem = Ref{Py}()
const desc = Ref{Py}()
const grappa = Ref{Py}()
  
function __init__()
    chem[] = pyimport("rdkit.Chem")
    desc[] = pyimport("rdkit.Chem.Descriptors")
    grappa[] = pyimport("grappa")

    MLPROP._GRAPPA[] = _GRAPPA_python
    MLPROP._get_descriptors[] = _get_descriptors_python
end

# GRAPPA
function _GRAPPA_python(components; userlocations, reference_state, verbose)
    components = CL.format_components(components)
    _params = CL.getparams(
        components, 
        CL.default_locations(GRAPPA); 
        userlocations, 
        ignore_missing_singleparams = ["Tc",],
        ignore_headers=["dipprnumber","inchikey","cas","canonicalsmiles","Pc","Vc","acentricfactor"]
    )
    
    _ABC = [pyconvert(Vector, grappa[].predict(s)) for s in _params["SMILES"].values]
    A = CL.SingleParam("A", components, [_abc[1] for _abc in _ABC])
    B = CL.SingleParam("B", components, [_abc[2] for _abc in _ABC])
    C = CL.SingleParam("C", components, [_abc[3] for _abc in _ABC])
    _T = Base.promote_eltype(A,B,C)
    
    params = MLPROP.GRAPPAParam(_params["Tc"],A,B,C)

    references = ["10.1016/j.ceja.2025.100750"]

    return GRAPPA(components, params, references)
end

# Get descriptors
function _get_descriptors_python(smiles::AbstractString)
    mol = chem[].MolFromSmiles(smiles)
    descs = Dict(
        "exactmw" => pyconvert(Float64, desc[].ExactMolWt(mol)),
        "NumRings" => pyconvert(Int64, desc[].RingCount(mol)),
        "NumHeteroatoms" => pyconvert(Int64, desc[].NumHeteroatoms(mol)),
        "NumHeavyAtoms" => pyconvert(Int64, desc[].HeavyAtomCount(mol)),
        "NumHBA" => pyconvert(Int64, desc[].NumHAcceptors(mol)),
        "NumHBD" => pyconvert(Int64, desc[].NumHDonors(mol)),
        "NumHalogens" => pyconvert(Int64, desc[].fr_halogen(mol)),
    )
    return isnothing(mol) ? nothing : descs
end


end