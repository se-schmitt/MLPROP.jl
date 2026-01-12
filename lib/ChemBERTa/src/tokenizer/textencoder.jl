abstract type AbstractTransformerTextEncoder <: AbstractTextEncoder end

"""
    struct TrfTextEncoder{
        T <: AbstractTokenizer,
        V <: AbstractVocabulary{String},
        C, A, EP, OP, DP, TP
    } <: AbstractTransformerTextEncoder
        tokenizer::T
        vocab::V
        config::C
        annotate::A
        process::EP
        onehot::OP
        decode::DP
        textprocess::TP
    end

The general text encoder. `TrfTextEncoder` has multiple fields that can modify the encode/decode process:

1. `.annotate` (default to `TextEncoders.annotate_strings`): Annotate the input string for the tokenizer,
 e.g. `String` would be treated as a single sentence, not a single word.
2. `.process` (default to `TextEncodeBase.nestedcall(TextEncoders.string_getvalue)`): The pre-process
 function applied to the tokenization results, e.g. adding special `end-of-sentence` token, computing attention mask...
3. `.onehot` (default to `TextEncoders.lookup_fist`): Apply onehot encoding on the preprocess result,
 the default behavior takes the first element from the proprocess result and applies onehot encoding.
4. `.decode` (default to `identity`): The function that converts each token id back to string. This can
 be used to handle some tokenizers that use a different set of vocabulary such as gpt2's byte-level vocabulary.
5. `.textprocess` (default to `TextEncodeBase.join_text`): the function that joins the `decode`-d result
 in complete sentence(s).

"""
struct TrfTextEncoder{
    T <: AbstractTokenizer,
    V <: AbstractVocabulary{String},
    C, A, EP, OP, DP, TP
} <: AbstractTransformerTextEncoder
    tokenizer::T
    vocab::V
    config::C
    annotate::A
    process::EP
    onehot::OP
    decode::DP
    textprocess::TP
end

"""
    TrfTextEncoder(
        tokenizer     :: AbstractTokenizer ,
        vocab         :: AbstractVocabulary{String} ,
        [ annotate    =  TextEncoders.annotate_string ,
        [ process     =  TextEncodeBase.nestedcall(TextEncoders.string_getvalue) ,
        [ onehot      =  TextEncoders.lookup_first ,
        [ decode      =  identity ,
        [ textprocess =  TextEncodeBase.join_text, ]]]]]
        ; config...)

Constructor of `TrfTextEncoder`. All keyword arguments are store in the `.config` field.
"""
TrfTextEncoder(tokenizer::AbstractTokenizer, vocab::AbstractVocabulary{String}; kws...) =
    TrfTextEncoder(tokenizer, vocab, annotate_strings; kws...)
TrfTextEncoder(tokenizer::AbstractTokenizer, vocab::AbstractVocabulary{String}, annotate; kws...) =
    TrfTextEncoder(tokenizer, vocab, annotate, nestedcall(string_getvalue); kws...)
TrfTextEncoder(tokenizer::AbstractTokenizer, vocab::AbstractVocabulary{String}, annotate, process; kws...) =
    TrfTextEncoder(tokenizer, vocab, annotate, process, lookup_first; kws...)
TrfTextEncoder(tokenizer::AbstractTokenizer, vocab::AbstractVocabulary{String}, annotate, process, onehot; kws...) =
    TrfTextEncoder(tokenizer, vocab, annotate, process, onehot, identity; kws...)
TrfTextEncoder(tokenizer::AbstractTokenizer, vocab::AbstractVocabulary{String}, annotate, process, onehot, decode;
               kws...) = TrfTextEncoder(tokenizer, vocab, annotate, process, onehot, decode, join_text; kws...)
function TrfTextEncoder(tokenizer::AbstractTokenizer, vocab::AbstractVocabulary{String},
                        annotate, process, onehot, decode, textprocess; kws...)
    return TrfTextEncoder(tokenizer, vocab, values(kws), annotate, process, onehot, decode, textprocess)
end

function Base.getproperty(e::TrfTextEncoder, sym::Symbol)
    if hasfield(TrfTextEncoder, sym)
        return getfield(e, sym)
    else
        return getfield(e, :config)[sym]
    end
end

@inline _membercall(f, e, x) = !(f isa Pipelines) && static_hasmethod(f, Tuple{typeof(e), typeof(x)}) ? f(e, x) : f(x)

TextEncodeBase.process(::Type{<:TrfTextEncoder}) = nestedcall(string_getvalue)
TextEncodeBase.tokenize(e::TrfTextEncoder, x) = getfield(e, :tokenizer)(_membercall(getfield(e, :annotate), e, x))
TextEncodeBase.process(e::TrfTextEncoder, x) = _membercall(getfield(e, :process), e, x)
TextEncodeBase.lookup(e::TrfTextEncoder, x) = _membercall(getfield(e, :onehot), e, x)
TextEncodeBase.decode(e::TrfTextEncoder, x) = _membercall(getfield(e, :decode), e, TextEncodeBase.decode_indices(e, x))
TextEncodeBase.decode_text(e::TrfTextEncoder, x) = _membercall(getfield(e, :textprocess), e, TextEncodeBase.decode(e, x))

TextEncodeBase.decode_indices(e::TrfTextEncoder, x) = decode_indices(e, x)
decode_indices(e::TrfTextEncoder, i::Union{Integer, OneHotArray, AbstractArray{<:Integer}}) =
    lookup(String, getfield(e, :vocab), i)
function decode_indices(e::TrfTextEncoder, x::AbstractArray)
    if ndims(x) < 2
        i = argmax(x)
    else
        amax = reshape(argmax(x; dims=1), Base.tail(size(x)))
        i = selectdim(reinterpret(reshape, Int, amax), 1, 1)
    end
    return decode_indices(e, i)
end

annotate_strings(x::AbstractString) = Sentence(x)
annotate_strings(x::Vector{<:AbstractString}) = Batch{Sentence}(x)
annotate_strings(x::Vector{<:Vector{<:AbstractString}}) = Batch{Batch{Sentence}}(x)
annotate_strings(x::Vector{<:Vector{<:Vector{<:AbstractString}}}) = Batch{Batch{Batch{Sentence}}}(x)

TextEncodeBase.onehot_encode(e::TrfTextEncoder, x) = lookup(OneHot, getfield(e, :vocab), x)

lookup_first(e::TrfTextEncoder, x) = TextEncodeBase.onehot_encode(e, x)
lookup_first(e::TrfTextEncoder, x::Tuple) = (TextEncodeBase.onehot_encode(e, x[1]), Base.tail(x)...)
function lookup_first(e::TrfTextEncoder, x::NamedTuple{name}) where name
    xt = Tuple(x)
    return NamedTuple{name}((TextEncodeBase.onehot_encode(e, xt[1]), Base.tail(xt)...))
end

# encoder constructor

const WList = Union{AbstractVector, AbstractVocabulary}

function TransformerTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary{String}, process,
                                startsym::String, endsym::String, padsym::String, trunc::Union{Nothing, Int})
    return TrfTextEncoder(
        tkr, vocab,
        @NamedTuple{startsym::String, endsym::String, padsym::String, trunc::Union{Nothing, Int}}(
            (startsym, endsym, padsym, trunc)),
        annotate_strings,
        process,
        lookup_first,
        identity,
        join_text,
    )
end

TransformerTextEncoder(tokenizef, v::WList, args...; kws...) =
    TransformerTextEncoder(WordTokenization(tokenize=tokenizef), v, args...; kws...)

TransformerTextEncoder(tkr::AbstractTokenizer, v::WList, args...; kws...) =
    throw(MethodError(TransformerTextEncoder, (tkr, v, args...)))

function TransformerTextEncoder(tkr::AbstractTokenizer, words::AbstractVector, process; trunc = nothing,
                                startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")
    vocab_list = copy(words)
    for sym in (padsym, unksym, startsym, endsym)
        sym ∉ vocab_list && push!(vocab_list, sym)
    end
    vocab = Vocab(vocab_list, unksym)
    return TransformerTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end

function TransformerTextEncoder(tkr::AbstractTokenizer, vocab::AbstractVocabulary, process; trunc = nothing,
                                startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")
    check_vocab(vocab, startsym) || @warn "startsym $startsym not in vocabulary, this might cause problem."
    check_vocab(vocab, endsym) || @warn "endsym $endsym not in vocabulary, this might cause problem."
    check_vocab(vocab, unksym) || @warn "unksym $unksym not in vocabulary, this might cause problem."
    check_vocab(vocab, padsym) || @warn "padsym $padsym not in vocabulary, this might cause problem."
    return TransformerTextEncoder(tkr, vocab, process, startsym, endsym, padsym, trunc)
end