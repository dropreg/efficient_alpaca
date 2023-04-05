SRC=src
TGT=tgt


DATA=/opt/data/private/data/llama/belle_1m
SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
MODEL=/opt/data/private/data/llama/tokenizer.model

python fsdp/scripts/utils/prepare_utils.py --manner split_zh --alpaca-data $DATA/Belle_open_source_1M.json

head -100 ${DATA}/train.src > ${DATA}/valid.src
head -100 ${DATA}/train.tgt > ${DATA}/valid.tgt

${SPM} --model=${MODEL} < ${DATA}/train.${SRC} > ${DATA}/train.spm.${SRC}.tmp
${SPM} --model=${MODEL} < ${DATA}/train.${TGT} > ${DATA}/train.spm.${TGT}.tmp
${SPM} --model=${MODEL} < ${DATA}/valid.${SRC} > ${DATA}/valid.spm.${SRC}.tmp
${SPM} --model=${MODEL} < ${DATA}/valid.${TGT} > ${DATA}/valid.spm.${TGT}.tmp

python fsdp/scripts/utils/prepare_utils.py --manner replace_zh --alpaca-data $DATA/Belle_open_source_1M.json

python fsdp/src/preprocess.py \
  --user-dir fsdp/src \
  --task llama_task \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/train.spm \
  --validpref ${DATA}/valid.spm \
  --destdir ${DATA}/data-bin \
  --srcdict alpaca/scripts/assert/dict.txt \
  --tgtdict alpaca/scripts/assert/dict.txt \
  --workers 40 \
