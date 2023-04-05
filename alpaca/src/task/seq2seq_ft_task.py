import torch
import itertools
import os
import logging
from typing import Dict, Optional

from dataclasses import dataclass, field
from fairseq import utils
from fairseq.tasks.translation import TranslationTask
from fairseq.utils import new_arange
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import TranslationConfig
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.data import iterators
from .dictionary import Dictionary
from .seq2seq_dataset import LanguagePairDataset
from fairseq.utils import safe_getattr, safe_hasattr

logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )

@dataclass
class FTTaskConfig(TranslationConfig):

    megatron_model: bool = field(
        default=False,
        metadata={"help": "using megatron-lm to split model"},
    )

    data_para: bool = field(
        default=False, metadata={"help": "data parallel"},
    )

@register_task("seq2seq_ft_task", dataclass=FTTaskConfig)
class Seq2SeqFineTuningTask(TranslationTask):

    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.data_para = safe_getattr(cfg, "data_para", False)
        self.megatron_model = safe_getattr(cfg, "megatron_model", False)
    
    def build_bpe(self, args):
        from sentencepiece import SentencePieceProcessor
        model_path = args.sentencepiece_model
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        return self.sp_model
    
    @classmethod
    def load_dictionary(cls, filename):
        if "dict.src.txt" not in filename or "dict.tgt.txt" not in filename:
            logger.info("{} is not exist!".format(filename))
            filename = "alpaca/scripts/assert/dict.txt"
            logger.info("load common dict {}!".format(filename))
        
        dictionary = Dictionary.load(filename)
        dictionary.pad_index = dictionary.add_symbol(dictionary.pad_word)
        return dictionary

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.cfg.data)
        data_path = paths[(epoch - 1) % len(paths)]
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.cfg.left_pad_source = False
        self.cfg.left_pad_target = False
        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            truncate_source=self.cfg.truncate_source,
            shuffle=(split != "test"),
            prepend_bos=True,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_generator(
        self,
        models,
        args=None,
        **kwargs,
    ):
        from generator.sequence_generator import SequenceGenerator
        from generator import search
        
        if isinstance(kwargs, dict):
            if "sampling" in kwargs:
                sampling = kwargs["sampling"] 
            else: 
                sampling = False
            if "sampling_topk" in kwargs: 
                sampling_topk = kwargs["sampling_topk"]
            else:
                sampling_topk = -1.0
            if "sampling_topp" in kwargs: 
                sampling_topp = kwargs["sampling_topp"]
            else:
                sampling_topp = -1.0
        else:
            sampling = getattr(args, "sampling", False)
            sampling_topk = getattr(args, "sampling_topk", -1.0)
            sampling_topp = getattr(args, "sampling_topp", -1.0)

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)
        
        extra_gen_cls_kwargs = {}
        return SequenceGenerator(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 512),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )
    
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            bos_token = sample['net_input']['bos_token']
            return generator.generate(
                models, sample, 
                prefix_tokens=prefix_tokens, constraints=constraints, 
                bos_token=bos_token,
            )

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        if not self.data_para:
            num_shards = 1
            shard_id=0
        return super().get_batch_iterator(
            dataset,
            max_tokens,
            max_sentences,
            max_positions,
            ignore_invalid_inputs,
            required_batch_size_multiple,
            seed,
            num_shards,
            shard_id,
            num_workers,
            epoch,
            data_buffer_size,
            disable_iterator_cache,
            skip_remainder_batch,
            grouped_shuffling,
            update_epoch_batch_itr,
        )