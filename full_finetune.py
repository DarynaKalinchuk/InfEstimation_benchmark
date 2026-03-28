from datasets import load_from_disk
from utils import get_preprocessed_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import argparse
import warnings
import os
import torch

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Full fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='TinyLlama', help='model name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--template', type=str, default='llama2', help='chat template')
    parser.add_argument('--val', action='store_true', default=False, help='whether to test on validation set')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--logging_step', type=int, default=10, help='logging step')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    args = parser.parse_args()

    os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"

    if args.model == 'TinyLlama':
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else:
        raise Exception("Invalid model name.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    if args.template == 'llama2':
        chat_template = "[INST] {prompt} [/INST] {response}"
    else:
        raise Exception("template options: [llama2]")

    dataset = load_from_disk("datasets/" + args.dataset)
    train_dataset = get_preprocessed_dataset(tokenizer, dataset['train'], chat_template, max_length=args.max_length)
    eval_dataset = get_preprocessed_dataset(tokenizer, dataset['test'], chat_template, max_length=args.max_length) if args.val else None
    save_path = f"finetuned_model/{args.model}/{args.dataset}_{args.epochs}"
    
    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_step,
        save_steps=10,
        save_total_limit=1,
        remove_unused_columns=False,
        learning_rate=2e-5,
        bf16=True,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model saved to: {save_path}")