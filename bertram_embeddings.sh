export BERTRAM_MODEL="models/bertram-add-for-bert-base-galician-cased"
export BERT_MODEL="models/bgl"
export OUTPUT_DIR="embedding-models/bgl-literal-embeddings"
export EXAMPLES_FOLDER="data/idiom-defs/gl/literal"
export NO_EXAMPLES=1

mkdir -p $OUTPUT_DIR

python3 bertram/train_embeddings.py \
   --bertram_model $BERTRAM_MODEL \
   --bert_model $BERT_MODEL \
   --output_dir $OUTPUT_DIR \
   --examples_folder $EXAMPLES_FOLDER \
   --no_examples $NO_EXAMPLES
