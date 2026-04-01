## Vocabulary
struct Vocab
    token_to_id::Dict{String,Int}
    unk_id::Int
end

function Vocab(tokens_json::AbstractDict{String}, unksym::String)
    token_to_id = Dict{String,Int}(token => Int(idx) + 1 for (token, idx) in tokens_json)
    return Vocab(token_to_id, token_to_id[unksym])
end

lookup(vocab::Vocab, token::AbstractString) = get(vocab.token_to_id, String(token), vocab.unk_id)

## Tokenizer
struct ChemBERTaTokenizer
    vocab::Vocab
    max_length::Int
    cls_id::Int
    sep_id::Int
    pad_id::Int
end

function load_tokenizer(::AbstractDict)
    vocab_json = JSON.parsefile(joinpath(DATADIR, "vocab.json"))
    vocab = Vocab(vocab_json, "[UNK]")
    return ChemBERTaTokenizer(
        vocab, 512,
        lookup(vocab, "[CLS]"),
        lookup(vocab, "[SEP]"),
        lookup(vocab, "[PAD]"),
    )
end

split_smiles(s::String) = [match.match for match in eachmatch(r"\[(.*?)\]|Br|Cl|.", s)]

function (tokenizer::ChemBERTaTokenizer)(smiles::AbstractString)
    tokens = split_smiles(String(smiles))
    # Build token IDs: [CLS] + tokens + [SEP]
    ids = Vector{Int}(undef, length(tokens) + 2)
    ids[1] = tokenizer.cls_id
    for (i, t) in enumerate(tokens)
        ids[i + 1] = lookup(tokenizer.vocab, t)
    end
    ids[end] = tokenizer.sep_id
    # Truncate if exceeding max length
    if length(ids) > tokenizer.max_length
        resize!(ids, tokenizer.max_length)
    end
    return ids
end
