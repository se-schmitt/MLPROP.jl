using PythonCall
using JLD2

script_dir = @__DIR__ 
project_path = normpath(joinpath(script_dir, "..", "..", "..", ".."))
# Path to Python weights
python_model_dir = joinpath(project_path, "test", "data", "python_multHANNA", "models", "HANNA", "ensemble")

# Output Path
database_path = joinpath(project_path, "database", "multHANNA")
out_path = joinpath(database_path, "parameters_states_all_multhanna.jld2")

torch = pyimport("torch")

all_ps = []
all_st = []

# Loop over all weight files and summarize in one file
for i in 0:9
    println("Reading parameter file $i...")
    pt_file = joinpath(python_model_dir, "HANNA_parameters_binary$i.pt")

    state_dict = torch.load(pt_file, map_location=torch.device("cpu"))
    
    # Help function to convert weights, bias and ci from PyTorch to Julia
    get_mat(name) = pyconvert(Matrix{Float32}, state_dict[name].detach().numpy())
    get_vec(name) = pyconvert(Vector{Float32}, state_dict[name].detach().numpy())
    get_sca(name) = Float32(pyconvert(Float64, state_dict[name].detach().item())) # Item for scalar

    # Layer 0 is Lipschitz
    theta_w  = get_mat("theta.0.linear.weight")
    theta_b  = get_vec("theta.0.linear.bias")
    theta_ci = [get_sca("theta.0.ci")]
    theta_u  = get_vec("theta.0._u")
    theta_v  = get_vec("theta.0._v")

    # Layer 0 and Layer 2 are Lipschitz
    alpha1_w  = get_mat("alpha.0.linear.weight")
    alpha1_b  = get_vec("alpha.0.linear.bias")
    alpha1_ci = [get_sca("alpha.0.ci")]
    alpha1_u  = get_vec("alpha.0._u")
    alpha1_v  = get_vec("alpha.0._v")

    alpha3_w  = get_mat("alpha.2.linear.weight")
    alpha3_b  = get_vec("alpha.2.linear.bias")
    alpha3_ci = [get_sca("alpha.2.ci")]
    alpha3_u  = get_vec("alpha.2._u")
    alpha3_v  = get_vec("alpha.2._v")

    # Layer 0 and Layer 2 are Lipschitz
    phi1_w  = get_mat("phi.0.linear.weight")
    phi1_b  = get_vec("phi.0.linear.bias")
    phi1_ci = [get_sca("phi.0.ci")]
    phi1_u  = get_vec("phi.0._u")
    phi1_v  = get_vec("phi.0._v")

    phi3_w  = get_mat("phi.2.linear.weight")
    phi3_b  = get_vec("phi.2.linear.bias")
    phi3_ci = [get_sca("phi.2.ci")]
    phi3_u  = get_vec("phi.2._u")
    phi3_v  = get_vec("phi.2._v")

    # build parameter and state namedtuples
	ps_i = (
        theta = (layer_1 = (weight = theta_w, bias = theta_b, ci = theta_ci), 
                layer_2 = NamedTuple()),
        
        alpha = (layer_1 = (weight = alpha1_w, bias = alpha1_b, ci = alpha1_ci), 
                layer_2 = NamedTuple(), 
                layer_3 = (weight = alpha3_w, bias = alpha3_b, ci = alpha3_ci), 
                layer_4 = NamedTuple()),
        
        phi   = (layer_1 = (weight = phi1_w, bias = phi1_b, ci = phi1_ci), 
                layer_2 = NamedTuple(), 
                layer_3 = (weight = phi3_w, bias = phi3_b, ci = phi3_ci))
    )

    st_i = (
        theta = (layer_1 = (u = theta_u, v = theta_v), 
                layer_2 = NamedTuple()),
        
        alpha = (layer_1 = (u = alpha1_u, v = alpha1_v), 
                layer_2 = NamedTuple(), 
                layer_3 = (u = alpha3_u, v = alpha3_v), 
                layer_4 = NamedTuple()),
        
        phi   = (layer_1 = (u = phi1_u, v = phi1_v), 
                layer_2 = NamedTuple(), 
                layer_3 = (u = phi3_u, v = phi3_v))
    )

    push!(all_ps, ps_i)
    push!(all_st, st_i)
end

# convert into fix tuples
final_ps = Tuple(all_ps)
final_st = Tuple(all_st)

jldsave(out_path; ps=final_ps, st=final_st)

println("Converting weights finished.")