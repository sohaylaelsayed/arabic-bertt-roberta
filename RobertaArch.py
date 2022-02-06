from transformers import RobertaConfig
from transformers import RobertaForMaskedLM


def initialize_roberta_model():

    config = RobertaConfig (
        attention_probs_dropout_prob = 0.1,
        hidden_act = "gelu",
        hidden_dropout_prob = 0.1,
        hidden_size = 768,
        initializer_range = 0.02,
        intermediate_size = 3072,
        layer_norm_eps = 1e-05,
        max_position_embeddings = 514,
        model_type = "roberta",
        num_attention_heads = 12,
        num_hidden_layers = 12,
        type_vocab_size = 1,
        vocab_size = 50265
    )


    model = RobertaForMaskedLM(config)
    return(model)