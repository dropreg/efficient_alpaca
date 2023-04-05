SRC=src
TGT=tgt

DATA=/opt/data/private/data/llama/llama_instruction/inf/
SPM=/opt/data/private/code/sentencepiece/build/src/spm_encode
MODEL=/opt/data/private/data/llama/tokenizer.model

cp ${DATA}/test.spm.${SRC} ${DATA}/test.spm.${TGT}

python alpaca/src/preprocess.py \
  --user-dir alpaca/src \
  --task llama_task \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --testpref ${DATA}/test.spm \
  --destdir ${DATA}/data-bin \
  --srcdict alpaca/scripts/assert/dict.txt \
  --tgtdict alpaca/scripts/assert/dict.txt \
  --workers 40 \
