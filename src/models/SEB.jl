
struct SEBParam{T}
    M::T
    b_ij::T
end

struct SEB{M,T}
    components::Vector{AbstractString}
    param::SEBParam{T}
    vis_model::M
end

"""
    SEB(SMILE_i, SMILE_j, vis_model)

## Description

SEB model for calculating diffusion coefficients at infinite dilution in a binary mixture.
The model predicts the boosting factor for the Stokes-Einstein equation using a neural network
trained on molecular descriptors of the solute and solvent.
"""
function SEB(SMILE_i::AbstractString, SMILE_j::AbstractString, eta_fun)
    #TODO use Clapeyron style
    # Loading weights and bias from nn_parameters.jld2
    path_nn_parameters = joinpath(DB_PATH, "SEB", "weights_bias_true.jld2")
    nn_parameters = load(path_nn_parameters)["Weights_Bias_SEB"]
    
    # Processing SMILES to get molecular descriptors used in neural net
    desc_i, desc_j = get_descriptors(SMILE_i), get_descriptors(SMILE_j)

    mw_i = desc_i["exactmw"] * 1e-3
    mw_j = desc_j["exactmw"] * 1e-3

    is_water_i = SMILE_i == "O" || SMILE_i == "[2H]O[2H]"
    is_water_j = SMILE_j == "O" || SMILE_j == "[2H]O[2H]"

    X_i_ini = [
        mw_i;
        is_water_i ? 0.5 : desc_i["NumHBA"] / desc_i["NumHeavyAtoms"];
        is_water_i ? 0.5 : desc_i["NumHBD"] / desc_i["NumHeavyAtoms"];
        desc_i["NumHeteroatoms"] / desc_i["NumHeavyAtoms"];
        desc_i["NumHalogens"] / desc_i["NumHeavyAtoms"];
        desc_i["NumRings"] != 0;
    ]
    X_j_ini = [
        mw_j;
        is_water_j ? 0.5 : desc_j["NumHBA"] / desc_j["NumHeavyAtoms"];
        is_water_j ? 0.5 : desc_j["NumHBD"] / desc_j["NumHeavyAtoms"];
        desc_j["NumHeteroatoms"] / desc_j["NumHeavyAtoms"];
        desc_j["NumHalogens"] / desc_j["NumHeavyAtoms"];
        desc_j["NumRings"] != 0;
    ]

    input = vcat(X_i_ini, X_j_ini)
    MW = mw_i
    b_ij_mean = 0.0

    #Initializing Neural Net using Lux
    NN = Chain(Dense(12 => 32, relu), Dense(32 => 16, relu), Dense(16 => 1, softplus))

    # Looping over parameter-sets to determine mean b_ij
    for (_, wb) in nn_parameters

        #Setting up the weights and bias of the neural net using Lux-Synatx
        st = (layer_1 = NamedTuple(), layer_2 = NamedTuple(), layer_3 = NamedTuple())
        ps = (
            (layer_1 = (weight = wb[2], bias = vec(wb[1]))),
            (layer_2 = (weight = wb[4], bias = vec(wb[3]))),
            (layer_3 = (weight = wb[6], bias = vec(wb[5]))),
        )
        #applying neural net with given weights and bias to calculate b_ij
        b_ij, st = NN(input, ps, st)
        b_ij_mean += only(b_ij)
    end
    b_ij_mean /= length(nn_parameters)

    # Constructing SEBParam-datastructure
    paramSEB = SEBParam(MW, b_ij_mean)

    return SEB([String(SMILE_i), String(SMILE_j)], paramSEB, eta_fun)
end

Base.broadcastable(x::SEB) = Ref(x)

"""
Diffusion
Diffusion calculates the diffusioncoefficent, using the Stokes-Einstein-equation and multiplying 
the result with the Boostingfactor
#Parameters
'model': Constructed SEB-model (see above)
'p': Pressure in Pa
'T': Temperature in Kelvin
"""
function Diffusion(model::SEB, p, T)
    # Initialitzing constants required for Stokes-Einstein-equation
    k_b = 1.380649e-23
    roh_i= 1050
    f=0.64
    M_i=model.param.M
    visc_j=model.vis_model(T)
    N_A=6.02214076e23
    r_i=((3*f*M_i)/(4*pi*roh_i*N_A))^(1/3)
    D_SEE_ij_infdil = (k_b*T)/(6*pi*visc_j*r_i)
    return D_SEE_ij_infdil * model.param.b_ij

end


export SEB,Diffusion