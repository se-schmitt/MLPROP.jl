# Load tokenizer
function load_tokenizer(config::AbstractDict)
    path_vocab_json = joinpath(DATADIR, "vocab.json")
    vocab = JSON.parsefile(path_vocab_json)
    idx_sort = sortperm(collect(values(vocab)))
    vocab_vector = collect(keys(vocab))[idx_sort]
    # vocab_vector = "[unused" .* string.(0:config["vocab_size"]-1) .* "]"
    # setindex!.(Ref(vocab_vector), collect(keys(vocab)), Int.(collect(values(vocab))).+1)
    encoder = TransformerTextEncoder(
        split_smiles, vocab_vector; 
        startsym="[CLS]", endsym="[SEP]", unksym="[UNK]", padsym="[PAD]", trunc=512
    )
    return ChemBERTaTokenizer(encoder)
end

split_smiles(s::String) = [match.match for match in eachmatch(r"\[(.*?)\]|Br|Cl|.", s)]

# ChemBERTaTokenizer
struct ChemBERTaTokenizer
    encoder
end
function (tokenizer::ChemBERTaTokenizer)(smiles)
    enc = encode(tokenizer.encoder, smiles)
    return getindex.(findall(enc.token),1)
end

# Base
struct TextTokenizer{T <: AbstractTokenization} <: AbstractTokenizer
    tokenization::T
end
TextTokenizer() = TextTokenizer(TextEncodeBase.DefaultTokenization())

TextEncodeBase.tokenization(tkr::TextTokenizer) = tkr.tokenization

