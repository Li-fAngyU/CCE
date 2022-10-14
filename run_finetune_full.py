# -*- coding: utf-8 -*-
# @Time    : 2020/4/25 22:59
# @Author  : Hui Wang

import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from trainers import FinetuneTrainer
from models import GRU4Rec, S3RecModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=0, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='Finetune_full', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument("--sample_num", type=int, default=1, help="sample num of rank loss ")
    parser.add_argument("--aggregation", type=str, default="mean", help="loss aggregation way")
    parser.add_argument("--rank_act", type=str, default="softmax", help="rank similarith act")
    parser.add_argument("--isfull", type=int, default=1, help="TODO")
    
    parser.add_argument("--istb", type=int, default=0, help="whether user tensorboard or not")
    parser.add_argument("--loss_type", type=str, default=None, help="loss type")
    parser.add_argument("--RQ3", type=int, default=0, help="loss type")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    if args.istb:
        from torch.utils.tensorboard import SummaryWriter
        #tb_file = 'log_tensorboard/' + args.model_name + args.loss_type +"/"
        args.output_dir = args.output_dir + args.model_name +'/' + args.data_name +'/'
        tb_file = args.output_dir + 'log_tensorboard/' + args.loss_type  +"/"
        writer = SummaryWriter(tb_file)
        args.writer = writer
    else:
        args.writer = None
    model_name = args.model_name
    args.model_name = args.model_name + '_' + args.loss_type


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.ckp}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    if 'GRU4Rec' == model_name:
        model = GRU4Rec(args=args)
    else:
        model = S3RecModel(args=args)

    trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)


    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        #pretrained_path = os.path.join(args.output_dir, f'{args.data_name}-epochs-{args.ckp}.pt')
        pretrained_path = os.path.join('reproduce/', f'{args.data_name}-epochs-{args.ckp}.pt')
        try:
            trainer.load(pretrained_path)
            print(f'Load Checkpoint From {pretrained_path}!')

        except FileNotFoundError:
            print(f'{pretrained_path} Not Found! The Model is same as {model_name}')

        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20

            #for introduction_result
            if args.RQ3 == 1:
                scores, _ = valid_train(trainer)
                if args.writer is not None:
                    writer.add_scalars("metric", {'train_HIT@5':scores[0],
                                                    'train_NDCG@5':scores[1],
                                                    'train_HIT@10':scores[2],
                                                    'train_NDCG@10':scores[3],
                                                    'train_HIT@20':scores[4],
                                                    'train_NDCG@20':scores[5]}, epoch)
            scores, _ = trainer.valid(epoch, full_sort=True)
            if args.writer is not None:
                writer.add_scalars("metric", {'HIT@5':scores[0],
                                                'NDCG@5':scores[1],
                                                'HIT@10':scores[2],
                                                'NDCG@10':scores[3],
                                                'HIT@20':scores[4],
                                                'NDCG@20':scores[5]}, epoch)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')

def valid_train(trainer):
    dataloader = trainer.train_dataloader
    epoch = 0
    trainer.model.eval()
    for i, batch in enumerate(dataloader):
        # 0. batch_data will be sent into the device(GPU or cpu)
        batch = tuple(t.to(trainer.device) for t in batch)
        user_ids, input_ids, target_pos, target_neg, answers, _ = batch
        answers = target_pos[:,-1].unsqueeze(-1)
        recommend_output = trainer.model.finetune(input_ids)

        recommend_output = recommend_output[:, -1, :]
        # 推荐的结果

        rating_pred = trainer.predict_full(recommend_output)

        rating_pred = rating_pred.cpu().data.numpy().copy()
        batch_user_index = user_ids.cpu().numpy()
        #rating_pred[trainer.args.train_matrix[batch_user_index].toarray() > 0] = 0
        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
        # 加负号"-"表示取大的值
        ind = np.argpartition(rating_pred, -20)[:, -20:]
        # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        # 对子表进行排序 得到从大到小的顺序
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        # 再取一次 从ind中取回 原来的下标
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if i == 0:
            pred_list = batch_pred_list
            answer_list = answers.cpu().data.numpy()
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
    return trainer.get_full_sort_score(epoch, answer_list, pred_list)

main()