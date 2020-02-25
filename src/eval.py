import os 
import sys 
from pathlib import Path
root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import torch


from src.utils.tokenization import BertTokenizer
from src.utils.optimization import BertAdam
from src.utils.bert_config import Config
from src.utils.data_reader import *
from src.models.bert_classifier import BertClassifier
from src.utils.cls_evaluate_funcs import acc_and_f1
from src.utils.bert_data_utils import convert_examples_to_features
from src.utils.bert_config import init_log


logging.basicConfig()
logger = logging.getLogger(__name__)

def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--config_path", default="", type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--task_name", default=None, type=str)
    # parser.add_argument("--output_dir", default=None,
    #     type=str, required=True, help="the outptu directory where the model predictions and checkpoints will")

    # # other parameters
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, help="the maximum total input sequence length after ")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--model_path", default='./', type=str)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3306)
    parser.add_argument("--nworkers", type=int, default=1)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_sign", type=str, default="")
    # classifier_sign == "single_linear
    parser.add_argument("--classifier_sign", type=str, default="single_linear")
    parser.add_argument("--log_path", type=str, default="")
    parser.add_argument("--vocab", type=str, default="")
    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def load_data(config):
 
    data_processor = Processor()

    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(config.vocab, do_lower_case=True)

    test_examples = data_processor.get_test_examples(config.data_dir)

    test_features = convert_examples_to_features(test_examples, label_list, config.max_seq_length, tokenizer,
                                                 task_sign=config.task_name)
    test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)

    test_dataloader = DataLoader(test_data, batch_size=config.test_batch_size, num_workers=config.nworkers)

    return test_dataloader, test_examples, label_list


def load_model(config, label_list):
    # device = torch.device(torch.cuda.is_available())
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    model = BertClassifier(config, num_labels=len(label_list))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare  optimzier
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    return model, device, n_gpu


def eval_checkpoint(model_object, eval_dataloader, config, \
                    device, n_gpu, label_list, eval_sign="dev"):
    model_object.eval()

    idx2label = {i: label for i, label in enumerate(label_list)}

    eval_loss = 0
    eval_accuracy = []
    predicts = []
    eval_f1 = []
    eval_recall = []
    eval_precision = []
    eval_steps = 0

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, label_ids)
            logits = model_object(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        input_mask = input_mask.to("cpu").numpy()
        logits = np.argmax(logits, axis=-1)

        input_mask = input_mask.tolist()
        eval_loss += tmp_eval_loss.mean().item()

        metric = acc_and_f1(preds=logits, labels=label_ids)
        predicts.extend(logits.tolist())
        eval_accuracy.append(metric['acc'])
        eval_precision.append(metric['precision'])
        eval_recall.append(metric['recall'])
        eval_f1.append(metric['f1'])
        eval_steps += 1

    average_loss = round(eval_loss / eval_steps, 4)
    eval_f1 = round(sum(eval_f1) / (len(eval_f1)), 4)
    eval_precision = round(sum(eval_precision) / len(eval_precision), 4)
    eval_recall = round(sum(eval_recall) / len(eval_recall), 4)
    eval_accuracy = round(sum(eval_accuracy) / len(eval_accuracy), 4)

    return average_loss, eval_accuracy, eval_f1, eval_precision, eval_recall, predicts


def merge_config(args_config):
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    test_loader, test_examples, label_list = load_data(config)
    model, device, n_gpu = load_model(config, label_list)
    state = torch.load(args_config.model_path + 'pytorch_model.bin', map_location='cuda')
    model.load_state_dict(state)
    model.eval()
    test_loss, test_acc, test_f1, test_prec, test_rec, predicts = eval_checkpoint(model,
                                                                    test_loader,
                                                                    config,
                                                                    device,
                                                                    n_gpu,
                                                                    label_list,
                                                                    eval_sign="test")
    with open(config.output_dir + 'results.txt', 'w') as f:
        for s, p in zip(test_examples, predicts):
            f.write(f'{s.text_a}\t{s.text_b}\t{s.label}\t{p}\n')
    print(f"TEST: precision: {test_prec}, recall: {test_rec},"
                f" f1: {test_f1}, acc: {test_acc}, loss: {test_loss}")

if __name__ == "__main__":
    main()
