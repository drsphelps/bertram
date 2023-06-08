export BERTRAM_MODEL="models/bertram-add-for-bert-base-uncased"
export BERT_MODEL="bert-base-uncased"
export OUTPUT_DIR="../models/bertram"
export EXAMPLES_FOLDER="../data/bertram"
export NO_EXAMPLES=10

python3 train_embeddings.py \
   --bertram_model $BERTRAM_MODEL \
   --bert_model $BERT_MODEL \
   --output_dir $OUTPUT_DIR \
   --examples_folder $EXAMPLES_FOLDER \
   --no_examples $NO_EXAMPLES
