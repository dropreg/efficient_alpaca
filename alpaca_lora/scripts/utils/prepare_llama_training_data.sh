SRC=src
TGT=tgt

DATA=/opt/data/private/data/llama/llama_raw
SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
MODEL=/opt/data/private/data/llama/tokenizer.model

python alpaca_lora/scripts/utils/prepare_utils.py --manner split --alpaca-data $DATA/alpaca_data.json

head -100 ${DATA}/train.src > ${DATA}/valid.src
head -100 ${DATA}/train.tgt > ${DATA}/valid.tgt

${SPM} --model=${MODEL} < ${DATA}/train.${SRC} > ${DATA}/train.spm.${SRC}.tmp
${SPM} --model=${MODEL} < ${DATA}/train.${TGT} > ${DATA}/train.spm.${TGT}.tmp
${SPM} --model=${MODEL} < ${DATA}/valid.${SRC} > ${DATA}/valid.spm.${SRC}.tmp
${SPM} --model=${MODEL} < ${DATA}/valid.${TGT} > ${DATA}/valid.spm.${TGT}.tmp

python alpaca_lora/scripts/utils/prepare_utils.py --manner replace --alpaca-data $DATA/alpaca_data.json

python alpaca_lora/src/preprocess.py \
  --user-dir alpaca_lora/src \
  --task llama_task \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/train.spm \
  --validpref ${DATA}/valid.spm \
  --destdir ${DATA}/data-bin \
  --srcdict alpaca_lora/scripts/assert/dict.txt \
  --tgtdict alpaca_lora/scripts/assert/dict.txt \
  --workers 40 \
