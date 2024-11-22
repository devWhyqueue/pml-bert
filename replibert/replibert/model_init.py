from transformers import BertModel


def initialize_from_hf_model(model, config):
    hs = config["hidden_size"]
    # Load Hugging Face model
    hf_model = BertModel.from_pretrained("bert-base-uncased")

    # Extract embeddings and encoder layer weights from the Hugging Face model, this is trivial do to similar architecture to hf embeddings
    model.embeddings.word_embeddings.weight.data = hf_model.embeddings.word_embeddings.weight.data.clone()
    model.embeddings.position_embeddings.weight.data = hf_model.embeddings.position_embeddings.weight.data.clone()
    model.embeddings.token_type_embeddings.weight.data = hf_model.embeddings.token_type_embeddings.weight.data.clone()
    model.embeddings.LayerNorm.weight.data = hf_model.embeddings.LayerNorm.weight.data.clone()
    model.embeddings.LayerNorm.bias.data = hf_model.embeddings.LayerNorm.bias.data.clone()

    # Initialize layers from the Hugging Face model
    for i,_ in enumerate(model.encoder):
        layer = model.encoder[i]
        hf_layer = hf_model.encoder.layer[i]

        # Attention weights, architecture from our model differs from hf model, considered
        layer.attention.in_proj_weight.data[:hs, :] = hf_layer.attention.self.query.weight.data.clone()
        layer.attention.in_proj_weight.data[hs:2*hs, :] = hf_layer.attention.self.key.weight.data.clone()
        layer.attention.in_proj_weight.data[2*hs:, :] = hf_layer.attention.self.value.weight.data.clone()
        layer.attention.out_proj.weight.data = hf_layer.attention.output.dense.weight.data.clone()  # the ouput projection layer, just one for both
        
        # Bias terms for attention layers
        layer.attention.in_proj_bias.data[:hs] = hf_layer.attention.self.query.bias.data.clone()
        layer.attention.in_proj_bias.data[hs:2*hs]= hf_layer.attention.self.key.bias.data.clone()
        layer.attention.in_proj_bias.data[2*hs:]  = hf_layer.attention.self.value.bias.data.clone()
        layer.attention.out_proj.bias.data = hf_layer.attention.output.dense.bias.data.clone()

        # Feed-forward network weights
        layer.intermediate.weight.data = hf_layer.intermediate.dense.weight.data.clone()
        layer.intermediate.bias.data = hf_layer.intermediate.dense.bias.data.clone()
        layer.output.weight.data = hf_layer.output.dense.weight.data.clone()
        layer.output.bias.data = hf_layer.output.dense.bias.data.clone()

        # Layer normalization weights, these are also learned in pretraining
        layer.attention_layer_norm.weight.data = hf_layer.attention.output.LayerNorm.weight.data.clone()
        layer.attention_layer_norm.bias.data = hf_layer.attention.output.LayerNorm.bias.data.clone()
        layer.output_layer_norm.weight.data = hf_layer.output.LayerNorm.weight.data.clone()
        layer.output_layer_norm.bias.data = hf_layer.output.LayerNorm.bias.data.clone()
    return hf_model