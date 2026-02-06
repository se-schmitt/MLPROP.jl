module PythonCallExt

using MLPROP
using PythonCall
const CL = MLPROP.Clapeyron

const grappa = Ref{Py}()
  
function __init__()
    grappa[] = pyimport("grappa")

    MLPROP._GRAPPA[] = _GRAPPA_python
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

end