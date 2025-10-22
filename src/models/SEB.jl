
function M(descs::Dict)
    return get(descs,"exactmw",0)
end
function R(descs::Dict)
    return !(get(descs,"NumRings",0) == 0)
end
function r_het(descs::Dict)
    return (get(descs,"NumHeteroatoms",0)/get(descs,"NumHeavyAtoms",0))
end
function r_acc(descs::Dict)
    return (get(descs,"NumHBA",0)/get(descs,"NumHeavyAtoms",0))
end
function r_don(descs::Dict)
    return (get(descs,"NumHBD",0)/get(descs,"NumHeavyAtoms",0))
end
function r_hal(SMILES::String,descs::Dict)
    mol = get_mol(SMILES)
    N_h=0
    for h in ["F", "Cl", "Br", "I"] # , "At", "Ts"]
       qmol = get_qmol(h)
       N_h += length(get_substruct_matches(mol, qmol))
    end
    return N_h/get(descs,"NumHeavyAtoms",0)
end


struct SEBParam
    "Molar mass in `kg/mol`"
    M
    "Correction factor for the Stokes-Einstein equation"
    b_ij
end

"""
SEB()


"""


struct SEB{M}
  components::Vector#{<AbstractString}
  param::SEBParam
  vis_model::M
end

function SEB(SMILE_i::String,SMILE_j::String)
    mol_i,mol_j = get_mol(SMILE_i), get_mol(SMILE_j)
    desc_i,desc_j = get_descriptors(mol_i), get_descriptors(mol_j)
    X_i_ini=[M(desc_i);R(desc_i);r_het(desc_i);r_hal(SMILE_i,desc_i);r_acc(desc_i);r_don(desc_i)]
    X_j_ini=[M(desc_j);R(desc_j);r_het(desc_j);r_hal(SMILE_j,desc_j);r_acc(desc_j);r_don(desc_j)]
    #b_ij Berechnung
end