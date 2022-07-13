export BERTRAM_MODEL="models/bertram-add-for-bert-base-english-uncased"
export BERT_MODEL="bert-base-uncased"
export OUTPUT_DIR="embedding-models/medical-terms-test/"
export EXAMPLES_FOLDER="data/cui_examples/processed"
export NO_EXAMPLES=150

python3 bertram/train_embeddings.py \
   --bertram_model $BERTRAM_MODEL \
   --bert_model $BERT_MODEL \
   --output_dir $OUTPUT_DIR \
   --examples_folder $EXAMPLES_FOLDER \
   --no_examples $NO_EXAMPLES
