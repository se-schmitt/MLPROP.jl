"""
    ChemBERTa.load()

Load the ChemBERTa model.
""" 
function load()
    path_cfg_json = joinpath(DATADIR,"config.json")
    config = JSON.parsefile(path_cfg_json)
    cfg = BERTConfig(; 
        vocab_size = config["vocab_size"],
        emb_dim = config["hidden_size"],
        max_pos_size = config["max_position_embeddings"],
        typ_voc_size = config["type_vocab_size"],
        n_heads = config["num_attention_heads"],
        n_layers = config["num_hidden_layers"],
        hidden_dim = config["intermediate_size"],
        dropout_rate = 0f0,
        act = NNlib.gelu_erf,
        dtype = Float32,
        position_offset = 2,
    )
    bert = BERT(cfg)

    tokenizer = load_tokenizer(config)
    
    ps, st = Lux.setup(rng, bert)
    path_sate_dict = joinpath(DATADIR, "model.safetensors")
    state_dict = load_safetensors(path_sate_dict)
    map_state_dict!(ps, state_dict)
    smodel = StatefulLuxLayer(bert, ps, Lux.testmode(st))

    return ChemBERTaModel(smodel, tokenizer)
end
