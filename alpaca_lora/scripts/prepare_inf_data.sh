SRC=src
TGT=tgt

DATA=/opt/data/private/data/nmt_data/llama_data/inference
SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
MODEL=/opt/data/private/data/llama/tokenizer.model

${SPM} --model=${MODEL} < ${DATA}/train.${SRC} > ${DATA}/train.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/train.${TGT} > ${DATA}/train.spm.${TGT}

python examples_nlg/llama/src/preprocess.py \
  --user-dir examples_nlg/llama/src \
  --task llama_translation \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/train.spm \
  --destdir ${DATA}/data-bin \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict /opt/data/private/data/llama/dict.txt \
  --tgtdict /opt/data/private/data/llama/dict.txt \
  --workers 40 \
