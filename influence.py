from datasets import load_from_disk
from transformers import AutoTokenizer
from utils import *
import pickle
import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning LLMs")
    parser.add_argument('--model', type=str, default='Llama-2-7b-chat-hf', help='model name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--hvp_cal', type=str, required=False, help='influence estimation method')
    parser.add_argument('--template', type=str, default='llama2', help='chat template')
    parser.add_argument('--max_length', type=int, default=128, help='tokenizer padding max length')
    parser.add_argument('--lambda_c', type=float, default=10, help='lambda const')
    parser.add_argument('--iter', type=int, default=3, help='#iteration')
    parser.add_argument('--alpha', type=float, default=1., help='alpha_const')
    parser.add_argument('--inf_args', type=str, required=False, help='Other args, method-specific.')
    args = parser.parse_args()

    # if 'Llama' in args.model:
    #     model_name = "/common/public/LLAMA2-HF/" + args.model
    # if args.model == 'mistral':
    #     model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
    if args.model == 'TinyLlama':
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    else: raise Exception("Invalid model name.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk("datasets/" + args.dataset)

    # results statistics directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_filename = f"{args.dataset}_{args.model}_{args.hvp_cal}_acc_cov.json".replace(" ", "_")
    metrics_path = os.path.join(results_dir, metrics_filename)


    if args.hvp_cal == "ekfac":

        influence_inf = pd.DataFrame(
        torch.load(f"results/EKFAC/{args.dataset}/{model_name}/influence_matrix.pt").numpy(),
        dtype=float
    )

    elif (args.hvp_cal == "random"):

        train_var = dataset["train"]["variation"]
        eval_var = dataset["test"]["variation"]
        N = len(train_var)
        #count of train samples per variation
        counts = {}
        for v in train_var:
            counts[v] = counts.get(v, 0) + 1

        # expected accuracy = expected coverage
        values = [counts[v] / N for v in eval_var]
        acc_rate = cov_rate = np.mean(values)

        metrics = {
        "accuracy": acc_rate,
        "coverage": cov_rate
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        sys.exit()



    elif "rep" in args.hvp_cal and "sim" in args.hvp_cal:

        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        model = PeftModel.from_pretrained(
            model,
            "lora_adapter/" + args.model + '/' + args.dataset + '_' + str(args.epochs)
        )
        model.eval()

        if args.template == 'llama2':
            chat_template = f"[INST] {{prompt}} [/INST]"
        else: raise Exception("template options: [llama2]")

        print('Generate hidden states...')

        check = []
        for p in tqdm(dataset['test']['prompts']):
            inputs = tokenizer(chat_template.format(prompt=p), padding=True, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            check.append(outputs['hidden_states'][-1][:, -1, :].view(-1).cpu().numpy().T)

        query = []
        for p in tqdm(dataset['train']['prompts']):
            inputs = tokenizer(chat_template.format(prompt=p), padding=True, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                query.append(outputs['hidden_states'][-1][:, -1, :].view(-1).cpu().numpy())


        sim_matrix = []
        for item in tqdm(check):
            arr = similarity_influence_estimation(test_vec=item, train_vecs=query,hvp_cal=args.hvp_cal)   # shape: [num_train]
            sim_matrix.append(arr)

        sim_df = pd.DataFrame(sim_matrix)

        influence_inf = -sim_df  #negation because the larger the similarity, the better; unlike influence.

    else:
        if args.template == 'llama2':
            chat_template = f"[INST] {{prompt}} [/INST] {{response}}"
        else: raise Exception("template options: [llama2]")

        os.makedirs('grad/' + args.model , exist_ok=True)
        core_path =  args.model + '/' + args.dataset + '_' + str(args.epochs)

        tr_grad_file = 'grad/' + core_path + '_tr.pkl'
        val_grad_file = 'grad/' + core_path + '_val.pkl'

        if os.path.exists(tr_grad_file) and os.path.exists(val_grad_file):
            with open(tr_grad_file, 'rb') as f:
                tr_grad_dict = pickle.load(f)
            with open(val_grad_file, 'rb') as f:
                val_grad_dict = pickle.load(f)
        else:
            print('collecting grad...')
            tokenized_tr = get_preprocessed_dataset(tokenizer, dataset['train'], chat_template, max_length=args.max_length)
            tokenized_val = get_preprocessed_dataset(tokenizer, dataset['test'], chat_template, max_length=args.max_length)
            tr_grad_dict, val_grad_dict = collect_gradient(model_name, "lora_adapter/" + core_path, tokenizer, tokenized_tr, tokenized_val)
            with open(tr_grad_file, 'wb') as f:
                pickle.dump(tr_grad_dict, f)
            with open(val_grad_file, 'wb') as f:
                pickle.dump(val_grad_dict, f)

        inf_args_map = dict(
        item.split('=') for item in (args.inf_args.split(',') if args.inf_args else [])
        )

        influence_inf = gradient_influence_estimation(tr_grad_dict, val_grad_dict, hvp_cal=args.hvp_cal, needed_args = inf_args_map)

    cache_dir = 'cache/' + args.model + '/'
    os.makedirs(cache_dir, exist_ok=True)
    influence_inf.to_csv(cache_dir + args.dataset + '_' + str(args.epochs) + args.hvp_cal + '.csv', index_label=False)
    check_acc_cov(influence = influence_inf, train_dataset = dataset['train'], 
                  validation_dataset = dataset['test'], 
                  metrics_path = metrics_path)


