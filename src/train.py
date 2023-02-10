import torch
import os
import logging
from src.config import set_args
from src.model import GPT2LMHeadModel
from transformers import BertTokenizer
from src.features import GPT2NewsTitleDataSet, collate_func
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from src.utils import set_seeds
from src.evaluate import evaluate


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, device, train_data, test_data, args):
    """
    训练模型
    Args:
        model: 模型
        device: 设备信息
        train_data: 训练数据类
        test_data: 测试数据类
        args: 训练参数配置信息

    Returns:

    """
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps参数无效，必须大于等于1")
    # 计算真实的训练batch_size大小
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func)
    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)
    logger.info("总训练步数为:{}".format(total_steps))
    model.to(device)
    # 获取模型所有参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    # 清空cuda缓存
    torch.cuda.empty_cache()

    title_id = train_data.title_id
    tr_loss, best_eval_loss = 0.0, float('inf')
    # 开始训练模型
    for iepoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        # 将模型调至训练状态
        model.train()
        iter_bar = tqdm(train_data_loader, desc="Iter (loss=X.XXX)", disable=False)
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            # 获取训练结果
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=input_ids, title_id=title_id)
            loss = outputs[0]
            tr_loss += loss.item()
            # 将损失值放到Iter中，方便观察
            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失进行回传
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 当训练步数整除累积步数时，进行参数优化
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        # 如果步数整除logging_steps，则记录学习率和训练集损失值
        # if args.logging_steps > 0 and (step + 1) % args.logging_steps == 0:
        logger.info(f'【Train】loss: {round(tr_loss/(len(train_data_loader)), 5)}')
        output_path = os.path.join(args.output_dir, 'model.pth')
        model_to_save = model.module if hasattr(model, "module") else model
        # epoch结束，进行模型测试，记录测试集的损失
        eval_loss = evaluate(model, device, test_data, args)
        logger.info(f'【Evaluate】loss: {round(eval_loss, 5)}')
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            # 每个epoch进行完，则保存模型
            torch.save(model_to_save.state_dict(), output_path)
            model_to_save.config.to_json_file(os.path.join(args.output_dir, 'config.json'))
            # 清空cuda缓存
            torch.cuda.empty_cache()


def main():
    # 设置模型训练参数
    args = set_args()
    # 设置显卡信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # 获取device信息，用于模型训练
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    # 设置随机种子，方便模型复现
    set_seeds(args.seed)
    # 实例化model
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
    # 实例化tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=True)
    # 将[space]作为一个分割整体，例如："我爱[Space]中国。"，使用原始tokenizer分词结果为"['我', '爱', '[', 'Space', ']', '中', '国', '。']";
    # 增加分割符号后的结果为"['我', '爱', '[Space]', '中', '国', '。']"
    tokens = ["[Space]", "[Content]", "[Title]"]
    tokenizer.add_tokens(tokens, special_tokens=True)
    model.transformer.resize_token_embeddings(len(tokenizer))
    # 创建模型的输出目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # 加载训练数据和测试数据
    train_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "train", args.train_file_path)
    test_data = GPT2NewsTitleDataSet(tokenizer, args.max_len, args.title_max_len, args.data_dir, "test", args.test_file_path)
    # 开始训练
    train(model, device, train_data, test_data, args)


if __name__ == '__main__':
    main()

