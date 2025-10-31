
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

The SEB-modell provides a way to calculate the diffusioncoefficents at infinite dilution in a binary
mixture utilizing a neural network to boost the results of the Stokes-Einstein-equation.

# Parameters

'components'

'param'

'vis_model'

# Constructor(?)


"""
struct SEB{M}
  components::Vector#{<AbstractString}
  param::SEBParam
  vis_model::M
end

function SEB(SMILE_i::String,SMILE_j::String,eta_fun)
    path_nn_parameters = joinpath(DB_PATH, "SEB", "nn_parameters.jld2")
    nn_parameters = load(path_nn_parameters)["Weights_Bias_SEB"]
    mol_i,mol_j = get_mol(SMILE_i), get_mol(SMILE_j)
    desc_i,desc_j = get_descriptors(mol_i), get_descriptors(mol_j)
    X_i_ini=[M(desc_i);R(desc_i);r_het(desc_i);r_hal(SMILE_i,desc_i);r_acc(desc_i);r_don(desc_i)]
    X_j_ini=[M(desc_j);R(desc_j);r_het(desc_j);r_hal(SMILE_j,desc_j);r_acc(desc_j);r_don(desc_j)]
    input=vcat(X_i_ini,X_j_ini)
    MW=M(desc_i)

    #b_ij Berechnung => Modell
    b_ij=0
    b_ij_sum=0
    NN = Chain(Dense(12 => 32, relu),Dense(32 => 16, relu),Dense( 16 => 1, softplus))
    #keys=["SEB_3";"SEB_7";"SEB_9";"SEB_12";"SEB_17";"SEB_19";"SEB_33";"SEB_42";"SEB_49";"SEB_55"]
    for (key,wb) in nn_parameters
        st_0=(layer_1=NamedTuple(),layer_2=NamedTuple(),layer_3=NamedTuple())
        ps=((layer_1=(weight=wb[2],bias=vec(wb[1]))),
        (layer_2=(weight=wb[4],bias=vec(wb[3]))),
        (layer_3=(weight=wb[6],bias=vec(wb[5]))))

        b_ij, st = NN(input,ps,st_0)
        b_ij_sum = b_ij_sum+b_ij[1]

    end
    b_ij_mean=b_ij/10
    #setup()

    paramSEB=SEBParam(MW,b_ij_mean)


    return SEB([SMILE_i;SMILE_j],paramSEB,eta_fun)
end

Base.broadcastable(x::SEB) = Ref(x)

function Diffusion(model::SEB,p::Float64,T::Float64)

k_b, roh_i, f, M_i,visc_j,N_A = 1.380649*10^(-23), 1050, 0.64, model.param.M, model.vis_model(T),6.02214076*10^23

D_SEE_ij_infdil = (k_b*T)/(6*pi*visc_j*(((3*f*M_i)/(4*pi*roh_i*N_A))^(1/3)))

return D_SEE_ij_infdil*model.param.b_ij

end


export SEB,Diffusion