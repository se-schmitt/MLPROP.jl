abstract type AbstractTransformerTextEncoder <: AbstractTextEncoder end

struct TrfTextEncoder{
    T <: AbstractTokenizer,
    V <: AbstractVocabulary{String},
    C, A, EP, OP
} <: AbstractTransformerTextEncoder
    tokenizer::T
    vocab::V
    config::C
    annotate::A
    process::EP
    onehot::OP
end

TrfTextEncoder(tokenizer::AbstractTokenizer, vocab::AbstractVocabulary{String}, annotate, process, onehot; kws...) =
    TrfTextEncoder(tokenizer, vocab, values(kws), annotate, process, onehot)

for name in fieldnames(TrfTextEncoder)
    (name == :tokenizer || name == :vocab) && continue
    @eval $(quote
        """
            set_$($(QuoteNode(name)))(builder, e::TrfTextEncoder)

        Return a new text encoder with the `$($(QuoteNode(name)))` field replaced with `builder(e)`.
        """
        function $(Symbol(:set_, name))(builder, e::TrfTextEncoder)
            setproperty!!(e, $(QuoteNode(name)), builder(e))
        end
    end)
end


TrfTextEncoder(builder, e::TrfTextEncoder) = set_process(builder, e)

function Base.getproperty(e::TrfTextEncoder, sym::Symbol)
    if hasfield(TrfTextEncoder, sym)
        return getfield(e, sym)
    else
        return getfield(e, :config)[sym]
    end
end

@inline _membercall(f, e, x) = !(f isa Pipelines) && static_hasmethod(f, Tuple{typeof(e), typeof(x)}) ? f(e, x) : f(x)

TextEncodeBase.tokenize(e::TrfTextEncoder, x) = getfield(e, :tokenizer)(_membercall(getfield(e, :annotate), e, x))
TextEncodeBase.process(e::TrfTextEncoder, x) = _membercall(getfield(e, :process), e, x)
TextEncodeBase.lookup(e::TrfTextEncoder, x) = _membercall(getfield(e, :onehot), e, x)
TextEncodeBase.onehot_encode(e::TrfTextEncoder, x) = lookup(OneHot, getfield(e, :vocab), x)

annotate_strings(x::AbstractString) = Sentence(x)

function lookup_first(e::TrfTextEncoder, x::NamedTuple{name}) where name
    xt = Tuple(x)
    return NamedTuple{name}((TextEncodeBase.onehot_encode(e, xt[1]), Base.tail(xt)...))
end

# encoder constructor
function TransformerTextEncoder(tokenizef, words; trunc = nothing, startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")
    fixedsize = false
    trunc_end = :tail
    pad_end = :tail
    
    tkr = TextTokenizer(WordTokenization(tokenize=tokenizef))
    vocab_list = copy(words)
    for sym in (padsym, unksym, startsym, endsym)
        sym ∉ vocab_list && push!(vocab_list, sym)
    end
    vocab = Vocab(vocab_list, unksym)
    enc = TrfTextEncoder(
        tkr, vocab,
        @NamedTuple{startsym::String, endsym::String, padsym::String, trunc::Union{Nothing, Int}}(
            (startsym, endsym, padsym, trunc)),
        annotate_strings,
        identity,
        lookup_first,
    )

    builder(e) = begin
        truncf = get_trunc_pad_func(e.padsym, fixedsize, e.trunc, trunc_end, pad_end)
        maskf = get_mask_func(e.trunc, :tail)
        # get token and convert to string
        Pipeline{:token}(nestedcall(string_getvalue), 1) |>
            # add start & end symbol
            Pipeline{:token}(with_head_tail(e.startsym, e.endsym), :token) |>
            # get mask with specific length
            Pipeline{:attention_mask}(maskf, :token) |>
            # truncate input that exceed length limit and pad them to have equal length
            Pipeline{:token}(truncf, :token) |>
            # convert to dense array
            Pipeline{:token}(nested2batch, :token) |>
            # sequence mask
            Pipeline{:sequence_mask}(identity, :attention_mask) |>
            # return token and mask
            PipeGet{(:token, :attention_mask, :sequence_mask)}()
    end

    return TrfTextEncoder(builder, enc)
end
