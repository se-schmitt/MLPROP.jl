import os
import torch
from transformers import AutoTokenizer, AutoModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from rdkit import Chem

def initialize_ChemBERTA(model_name="DeepChem/ChemBERTa-77M-MTR", device=None):

    model_path = '../../lib/ChemBERTa/data'

    ChemBERTA = AutoModel.from_pretrained(
        model_path,
        local_files_only=True
    ).to(device)

    # Check if the vocabulary file already exists
    vocab_path = os.path.join(model_path, 'vocab.json')
    if not os.path.exists(vocab_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
    # Load custom tokenizer using the saved vocab.json
    custom_tokenizer = Tokenizer(
        WordLevel.from_file(
            vocab_path,  # Path to your custom vocabulary file
            unk_token='[UNK]'
        )
    )

    # Set the pre-tokenizer to split SMILES characters (including handling Br, Cl, etc.)
    pre_tokenizer = Split(
        pattern=Regex(r"\[(.*?)\]|Br|Cl|."),
        behavior='isolated'
    )
    custom_tokenizer.pre_tokenizer = pre_tokenizer
    return ChemBERTA, custom_tokenizer

def get_smiles_embedding(smiles, custom_tokenizer, ChemBERTA, device, max_length=512):

    # canonicalize the SMILES
    smiles = canonicalize_smiles(smiles)

    # Tokenize the SMILES using your custom tokenizer
    custom_encoded = custom_tokenizer.encode(smiles)

    # Add [CLS] and [SEP] tokens
    CLS_token_id = 12  # Assuming 12 is the token ID for [CLS]
    SEP_token_id = 13  # Assuming 13 is the token ID for [SEP]
    PAD_token_id = 0   # Assuming 0 is the token ID for [PAD]

    # Create input_ids with [CLS] at the start and [SEP] at the end
    input_ids = [CLS_token_id] + custom_encoded.ids + [SEP_token_id]

    # Apply truncation if input_ids are longer than max_length
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length - 1] + [SEP_token_id]  # Ensure the sequence ends with [SEP]

    # Apply padding if input_ids are shorter than max_length
    if len(input_ids) < max_length:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [PAD_token_id] * padding_length

    # Convert the input_ids to PyTorch tensor format
    input_ids_tensor = torch.tensor([input_ids]).to(device)

    # Prepare attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids_tensor != PAD_token_id).long()

    with torch.no_grad():
        # Get embeddings from ChemBERTA
        emb = ChemBERTA(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask
        )["last_hidden_state"][:, 0, :].cpu().numpy()# Take [CLS] token embedding
    return emb

def canonicalize_smiles(smiles):
    # Canonicalize the SMILES
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return smiles