@concrete struct LipschitzDense <: LuxCore.AbstractLuxLayer
    activation
    in_dims<:Lux.IntegerType
    out_dims<:Lux.IntegerType
    init_weight
    init_bias
    init_ci
    eps::Real
end

function Base.show(io::IO, d::LipschitzDense)
    print(io, "LipschitzDense($(d.in_dims) => $(d.out_dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    return print(io, ")")
end

function LipschitzDense(in_dims::Integer, out_dims::Integer, act; 
        init_weight=glorot_uniform, init_bias=zeros32, init_ci=rng->rand(rng,Float32,1)*20, eps=1f-12)
    return LipschitzDense(act, in_dims, out_dims, init_weight, init_bias, init_ci, eps)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::LipschitzDense)
    return (;
        weight = l.init_weight(rng, l.out_dims, l.in_dims), 
        bias = l.init_bias(rng, l.out_dims),
        ci = l.init_ci(rng)
    )
end

function LuxCore.initialstates(rng::AbstractRNG, l::LipschitzDense)
    return (;
        warmstart = Val(false),  
        training = Val(true), 
        u = randn(rng, Float32, l.out_dims), 
        v = randn(rng, Float32, l.in_dims)
    )
end

LuxCore.parameterlength(d::LipschitzDense) = d.out_dims * d.in_dims + d.out_dims + 1
LuxCore.statelength(d::LipschitzDense) = 4
LuxCore.outputsize(d::LipschitzDense, _, ::AbstractRNG) = (d.out_dims,)

function (l::LipschitzDense)(x::AbstractArray, ps, st::NamedTuple)
    if iswarmstart(st)
        weight = ps.weight
    else
        LuxOps.istraining(st) ? power_iteration!(st.u, st.v, deepcopy(ps.weight)) : nothing
        largest_sv = dot(st.u, ps.weight * st.v)        # detach here?
        weight = (ps.weight / (largest_sv+l.eps)) * softplus(ps.ci[1])
    end
    _x = Lux.Utils.make_abstract_matrix(x)
    y = Lux.Utils.matrix_to_array(
        fused_dense_bias_activation(l.activation, weight, _x, ps.bias), x
    )
    return y, st
end


# @concrete struct LipschitzLinear <: AbstractLuxLayer
#     in_dims::Int
#     out_dims::Int
#     n_power_iterations::Int
# end

# function LipschitzLinear(in_dims::Int, out_dims::Int; n_power_iterations=2)
#     return LipschitzLinear(in_dims, out_dims, n_power_iterations)
# end

# # initial
# function Lux.initialparameters(rng::AbstractRNG, l::LipschitzLinear)
#     return (
#         # W_raw random initialization 
#         weight = randn(rng, Float32, l.out_dims, l.in_dims) .* 0.1f0, 
#         bias = zeros(Float32, l.out_dims),
#         # ci starts at 4.0
#         ci = [4.0f0] 
#     )
# end

# function Lux.initialstates(rng::AbstractRNG, l::LipschitzLinear)
#     u_init = randn(rng, Float32, l.out_dims)
#     v_init = randn(rng, Float32, l.in_dims)
    
#     return (
#         u = u_init ./ norm(u_init),
#         v = v_init ./ norm(v_init)
#     )
# end

# function ((l::LipschitzLinear)(x, ps, st))
#     W = ps.weight
#     u = st.u
#     v = st.v
    
#     # # Power Iteration (only for training)
#     # for _ in 1:l.n_power_iterations
#     #     v_new = W' * u
#     #     v = v_new ./ (norm(v_new) + 1e-12f0) 
        
#     #     u_new = W * v
#     #     u = u_new ./ (norm(u_new) + 1e-12f0)
#     # end
    
#     largest_sv = dot(u, W * v)
    
#     W_normed = W ./ (largest_sv + 1e-12)
    
#     softplus_ci = log(1.0f0 + exp(ps.ci[1]))
    
#     W_scaled = W_normed .* softplus_ci
    
#     y = W_scaled * x .+ ps.bias
    
#     # state doesnt change
#     #st_new = (u = u, v = v)
    
#     return y, st
# end