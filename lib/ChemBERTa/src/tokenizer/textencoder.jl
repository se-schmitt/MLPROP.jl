struct TransformerTextEncoder{
    T <: AbstractTokenizer,
    V <: AbstractVocabulary{String},
    C, A, EP, OP
} <: AbstractTextEncoder
    tokenizer::T
    vocab::V
    config::C
    annotate::A
    process::EP
    onehot::OP
end

TransformerTextEncoder(tokenizer::AbstractTokenizer, vocab::AbstractVocabulary{String}, annotate, process, onehot; kws...) =
    TransformerTextEncoder(tokenizer, vocab, values(kws), annotate, process, onehot)

function Base.getproperty(e::TransformerTextEncoder, sym::Symbol)
    if hasfield(TransformerTextEncoder, sym)
        return getfield(e, sym)
    else
        return getfield(e, :config)[sym]
    end
end

@inline _membercall(f, e, x) = !(f isa Pipelines) && static_hasmethod(f, Tuple{typeof(e), typeof(x)}) ? f(e, x) : f(x)

TextEncodeBase.tokenize(e::TransformerTextEncoder, x) = getfield(e, :tokenizer)(_membercall(getfield(e, :annotate), e, x))
TextEncodeBase.process(e::TransformerTextEncoder, x) = _membercall(getfield(e, :process), e, x)
TextEncodeBase.lookup(e::TransformerTextEncoder, x) = _membercall(getfield(e, :onehot), e, x)
TextEncodeBase.onehot_encode(e::TransformerTextEncoder, x) = lookup(OneHot, getfield(e, :vocab), x)

annotate_strings(x::AbstractString) = Sentence(x)

function lookup_first(e::TransformerTextEncoder, x::NamedTuple{name}) where name
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
    enc = TransformerTextEncoder(
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

    return setproperty!!(enc, :process, builder(enc))
end
