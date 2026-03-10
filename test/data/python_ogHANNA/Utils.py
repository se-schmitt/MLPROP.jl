import torch
import numpy as np
from rdkit import Chem


def predict(embedding_matrix, scaler, model, device):
    """Predict ln(gamma) values from an embedding matrix."""
    processed = preprocess_input(embedding_matrix)
    scaled = scaler.transform(processed)
    temp, x1, fp = split_and_reshape_input(scaled)

    temp_tensor = torch.tensor(temp, dtype=torch.float32).to(device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32).to(device)
    fp_tensor = torch.tensor(fp, dtype=torch.float32).to(device)

    ln_gammas_pred, _ = model(temp_tensor, x1_tensor, fp_tensor)
    return x1_tensor.detach().cpu().numpy(), ln_gammas_pred.detach().cpu().numpy()


def preprocess_input(embedding_matrix, Embedding_BERT=384, num_components=2):
    num_samples = embedding_matrix.shape[0]
    expected_columns = 1 + (num_components - 1) + num_components * Embedding_BERT
    assert embedding_matrix.shape[1] == expected_columns, (
        f"Input shape doesn't match the expected shape based on Embedding_BERT. "
        f"Expected {expected_columns} columns, got {embedding_matrix.shape[1]}"
    )

    temp = embedding_matrix[:, :1]
    reshaped_data = np.zeros((num_samples, num_components, Embedding_BERT + 2))
    reshaped_data[:, :, 0] = temp[:, 0, None]

    for i in range(num_components):
        if i != num_components - 1:
            reshaped_data[:, i, 1] = embedding_matrix[:, i + 1]
        else:
            reshaped_data[:, i, 1] = 1 - np.sum(reshaped_data[:, :-1, 1], axis=1)

        start_idx = 1 + num_components - 1 + i * Embedding_BERT
        end_idx = start_idx + Embedding_BERT
        reshaped_data[:, i, 2:] = embedding_matrix[:, start_idx:end_idx]

    return reshaped_data


def split_and_reshape_input(input_array):
    standardized_temp = input_array[:, 0, 0]
    mole_fractions_n_minus_1 = input_array[:, 0, 1:2]
    feature_points = input_array[:, :, 2:]
    return standardized_temp, mole_fractions_n_minus_1, feature_points


def canonicalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def get_smiles_embedding(smiles, custom_tokenizer, ChemBERTA, device, max_length=512):
    custom_encoded = custom_tokenizer.encode(smiles)

    cls_token_id = 12
    sep_token_id = 13
    pad_token_id = 0

    input_ids = [cls_token_id] + custom_encoded.ids + [sep_token_id]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length - 1] + [sep_token_id]

    if len(input_ids) < max_length:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [pad_token_id] * padding_length

    input_ids_tensor = torch.tensor([input_ids]).to(device)
    attention_mask = (input_ids_tensor != pad_token_id).long()

    with torch.no_grad():
        emb = ChemBERTA(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask
        )["last_hidden_state"][:, 0, :].cpu().numpy()
    return emb


def create_embedding_matrix(smiles1, smiles2, T, device, ChemBERTA, custom_tokenizer, x1_values=None):
    smiles1 = canonicalize_smiles(smiles1)
    smiles2 = canonicalize_smiles(smiles2)

    emb1 = get_smiles_embedding(smiles1, custom_tokenizer=custom_tokenizer, ChemBERTA=ChemBERTA, device=device).flatten()
    emb2 = get_smiles_embedding(smiles2, custom_tokenizer=custom_tokenizer, ChemBERTA=ChemBERTA, device=device).flatten()

    if x1_values is None:
        x1_values = np.linspace(0, 1, 100)

    embedding_matrix = []
    for x1 in x1_values:
        row = np.concatenate(([T, x1], emb1, emb2))
        embedding_matrix.append(row)

    return np.array(embedding_matrix)



