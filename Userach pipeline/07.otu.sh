#Before the final step of clustering, all sequences will have their corresponding sample plot numbers added before the sequence ID after the previous step of Unique processing. Insect sequences are filtered and merged through deep learning models, while plant sequences are directly merged.
#!/bin/bash
cd "your working path"
USEARCH_PATH="~/soft/usearch"
INPUT_DIR="06.Unique"
OUTPUT_DIR="07.OTU"
mkdir -p "$OUTPUT_DIR"
"$USEARCH_PATH" -sortbysize "$INPUT_DIR"/all.uni.fa -fastaout "$INPUT_DIR"/all.sort.fa -minsize 2
"$USEARCH_PATH" -cluster_otus "$INPUT_DIR"/all.sort.fa -otus "$OUTPUT_DIR"/cluster/all.otus.fa -relabel OTU -uparseout "$OUTPUT_DIR"/cluster/all_out -minsize 2
