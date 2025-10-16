# Quality control of merged reads
#!/bin/bash
cd "your working path"
USEARCH_PATH="~/soft/usearch"

OUTPUT_DIR="04.Maxee"
mkdir -p "$OUTPUT_DIR_FILTERED"

for merged_file in 03.Merged/*_merged.fq; do
    filtered_file="$OUTPUT_DIR/$(basename "${merged_file/_merged.fq/.MAXEE_1.fasta}")"
    "$USEARCH_PATH" -fastq_filter "$merged_file" -fastaout "$filtered_file" --fastq_maxee 1
    echo "Filtered and converted: $merged_file -> $filtered_file"
done
