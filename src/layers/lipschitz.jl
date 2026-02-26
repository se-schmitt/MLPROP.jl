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
