from transformers import LlamaConfig, LlamaModel, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig, AqlmConfig
from transformers import GemmaModel, GemmaConfig
# from transformers import GemmaConfig
import transformers
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training


class LLM_PEFT(nn.Module):

    def __init__(self, configs,input_dim, output_dim,LoRA_alpha,LoRA_rank,model_name, device):
        super(LLM_PEFT, self).__init__()
        self.num_hidden_layers = configs.num_hidden_layers
        self.device = device

        quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    # llm_int8_has_fp16_weight = True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                    # ,llm_int8_skip_modules =['layers']
                )
        
        

        if configs.model_type == 'gemma':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            # model_name = 'google/gemma-2b-it'
            self.gemma_config = GemmaConfig.from_pretrained(model_name)
            self.gemma_config.output_attentions = True
            self.gemma_config.output_hidden_states = True
            self.gemma_config.attention_bias = False


            
            try:
                self.llm_model = GemmaModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gemma_config,
                    # device_map="auto",
                    device_map="cuda:0",
                    quantization_config = quantization_config

                )
                
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GemmaModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gemma_config,
                    # device_map="auto",
                    device_map="cuda:0",
                    quantization_config = quantization_config
                )
            print(self.llm_model)
            # change the input features of the origin embedding layers
            embedding_dim = self.llm_model.embed_tokens.embedding_dim
            # self.llm_model.embed_tokens = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
            self.llm_model.embed_tokens = nn.Identity()

            # Enable Gradient Checkpointing
            self.llm_model.config.use_cache = False 
            self.llm_model.gradient_checkpointing_enable()
            self.llm_model = prepare_model_for_kbit_training(self.llm_model, use_gradient_checkpointing=True)  

            # bias: Bias type for Lora. Can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training. Be aware that this means that, even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation.
            # task_type: one of {SEQ_CLS, TOKEN_CLS, CAUSAL_LM, SEQ_2_SEQ_LM, QUESTION_ANS, FEATURE_EXTRACTION}
            peft_config = LoraConfig(
                lora_alpha=LoRA_alpha, # for Lora scali ng
                lora_dropout=0.1, # The dropout probability for Lora layers
                target_modules=['k_proj','q_proj'],
                r=LoRA_rank, # the rank of the update matrices. Lower rank results in smaller update matrices with fewer trainable parameters.
                bias="none",
                # task_type="SEQ_2_SEQ_LM"
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            print('----------PEFT Model----------')
            print(self.llm_model)


            # set embedding layer to trainable
            for name, param in self.llm_model.named_parameters():
                if 'embed_tokens' in str(name):
                    # print(f'Unfreezing parameter {name}')
                    param.requires_grad = True


            # trainable, total = self.llm_model.get_nb_trainable_parameters()
            # print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")
            # print(f'Memory used by model: {round(self.llm_model.get_memory_footprint()/1024/1024/1024, 2)} GB')
            
                
    def forward(self,inputs_embeds):
        
        # output_embeds =self.llm_model(inputs_embeds=inputs_embeds).last_hidden_state
        output_embeds =self.llm_model(inputs_embeds=inputs_embeds).last_hidden_state
        return output_embeds
        

if __name__ == "__main__":
    # device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    GPU_name = torch.cuda.get_device_name()

    transformers.logging.set_verbosity_error()
    model_name = 'google/gemma-2b-it'
    config = GemmaConfig.from_pretrained(model_name)
    
    input_dim = 700
    model = LLM_LoRA(config,input_dim,device)
    # print(model)
    # print(config)
    model.LoRA()