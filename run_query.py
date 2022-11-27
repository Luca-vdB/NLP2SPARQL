# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (BERT, RoBERTa).
BERT and RoBERTa are fine-tuned using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import

import argparse
import logging
import os
import random
import re
from io import open

import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, BertConfig, BertModel, BertTokenizer)

from model import Seq2Seq, BertSeq2Seq

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 'bert': (BertConfig, BertModel, BertTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target
    
    def __str__(self) -> str:
        return self.source, self.target


def read_examples(query_file, question_file):
    """Read examples from filename."""
    examples = []
    with open(query_file, encoding="utf-8") as query_f:
        with open(question_file, encoding='utf-8') as question_f:
            for idx, (query, question) in enumerate(zip(query_f, question_f)):
                #print(question.strip(), query.strip())
                examples.append(
                    Example(
                        idx=idx,
                        source=question.strip(),
                        target=query.strip(),
                    )
                )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_architecture", default=None, type=str, required=True,
                        help="Model architecture: e.g. bert2bert, bert2rnd")
    parser.add_argument("--encoder_model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--decoder_model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--source", default="sparql", type=str,
                        help="The source language (for file extension)")
    parser.add_argument("--target", default="en", type=str,
                        help="The target language (for file extension)")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--save_inverval', type=int, default=1,
                        help="save checkpoint every N epochs")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.encoder_model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.encoder_model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    # budild model
    encoder = model_class.from_pretrained(args.encoder_model_name_or_path, config=config)
    if args.model_architecture == 'bert2rnd':
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    elif args.model_architecture == 'bert2bert':
        decoder_config = config_class.from_pretrained(args.config_name if args.config_name else args.decoder_model_name_or_path)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder = model_class.from_pretrained(args.decoder_model_name_or_path, config=decoder_config)
        model = BertSeq2Seq(encoder=encoder, decoder=decoder, config=config,
                            beam_size=args.beam_size, max_length=args.max_target_length,
                            sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    else:
        raise Exception("Model architecture is not valid.")

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))

        for key in list(model.state_dict().keys()):
            model.state_dict()[key.replace('bert', 'encoder')] = model.state_dict().pop(key)

        model.load_state_dict(torch.load(args.load_model_path))  # Removed  strict=False, need to fix this!

    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_test:
        files = []
        if args.dev_filename is not None:
            files.append(args.dev_filename)
        if args.test_filename is not None:
            #files.append(args.test_filename)
            pass
        for idx, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file + "." + args.source, file + "." + args.target)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
            all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_source_ids, all_source_mask)

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    preds = model(source_ids=source_ids, source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)
                        print(text)
            model.train()
            predictions = []
            pred_str = []
            label_str = []
            with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx))), 'w') as f, open(
                    os.path.join(args.output_dir, "test_{}.gold".format(str(idx))), 'w') as f1:
                for ref, gold in zip(p, eval_examples):
                    ref = ref.strip().replace('< ', '<').replace(' >', '>')
                    ref = re.sub(r' ?([!"#$%&\'(’)*+,-./:;=?@\\^_`{|}~]) ?', r'\1', ref)
                    ref = ref.replace('attr_close>', 'attr_close >').replace('_attr_open', '_ attr_open')
                    ref = ref.replace(' [ ', ' [').replace(' ] ', '] ')
                    ref = ref.replace('_obd_', ' _obd_ ').replace('_oba_', ' _oba_ ')

                    pred_str.append(ref.split())
                    label_str.append([gold.target.strip().split()])
                    predictions.append(str(gold.idx) + '\t' + ref)
                    f.write(str(gold.idx) + '\t' + ref + '\n')
                    f1.write(str(gold.idx) + '\t' + gold.target + '\n')
            bl_score = corpus_bleu(label_str, pred_str) * 100
            logger.info("  %s = %s " % ("BLEU", str(round(bl_score, 4))))
            logger.info("  " + "*" * 20)


if __name__ == "__main__":
    main()
