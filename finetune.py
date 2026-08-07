import os
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
import argparse
import warnings
import torch
import random
import numpy as np
from huggingface_hub import login
warnings.filterwarnings("ignore")
import sys
import shutil

seed = 1
print(f"Setting random seed: {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

from utils import *

with open("settings_txt/TOKENS.txt", "r") as f:
    line = f.read().strip()

login(token=line.split("=", 1)[1].strip().strip('"'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Olmo', help='model name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--lora_r', type=int, default=4, help='lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha')
    args = parser.parse_args()
    
    os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"

    target_modules=[
        "q_proj",
        # "k_proj",
        "v_proj",
        # "o_proj",
    ]
    
    model_name = get_model_name(args.model)


    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=None,
            device_map='auto'
    )
    model.config.use_cache = False

    print(f"Model {model_name} loaded successfully.")

    save_path = f"lora_adapter/{args.model}/{args.dataset}_{args.epochs}"
    # Deleting files in that dir if exist, not to accidentally take old checkpoint results			
    if os.path.isdir(save_path):			
        shutil.rmtree(save_path)

    for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        os.environ.pop(var, None)

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    
    dataset = load_from_disk("datasets/" + args.dataset)
    train_dataset = get_preprocessed_dataset(tokenizer, dataset['train'], max_length=args.max_length)  
    eval_dataset = get_preprocessed_dataset(tokenizer, dataset['test'], max_length=args.max_length)
    print(f"Training {args.model} for {args.epochs} epochs with batch size {args.batch_size}")

 

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=10, # max number of checkpoints
        remove_unused_columns=False,
        learning_rate = 5e-5,
        seed=seed,
        data_seed=seed,
        full_determinism=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    trainer.train()
    
    
    print("Training completed.")
    trainer.save_model(save_path)
    print(f"Model saved to: {save_path}")
    
