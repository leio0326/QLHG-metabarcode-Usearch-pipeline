#Process all sequences of appropriate length to ensure that they are unique
#!/bin/bash
cd "your working path"
USEARCH_PATH="~/soft/usearch"
INPUT_DIR="05.length"
OUTPUT_DIR="06.Unique"
mkdir -p "$OUTPUT_DIR"
for input_file1 in "$INPUT_DIR"/*.fa; do
    file_prefix="${input_file1%.fa}"     
    "$USEARCH_PATH"  -fastx_uniques $input_file1 -fastaout "$OUTPUT_DIR"/${file_prefix}.uni.fa -sizeout 
done

