# 03.Megerd all PE reads
#!/bin/bash
cd "your working path"
#Your Usearch path
USEARCH_PATH="~/soft/usearch"
INPUT_DIR="02.Cutprimer"
OUTPUT_DIR="03.Merged"
mkdir -p "$OUTPUT_DIR"
for file1 in "$INPUT_DIR"/*_1.fq; do
    file2="${file1/_1.fq/_2.fq}"
    if [[ -f "$file2" ]]; then
        output_file="${OUTPUT_DIR}/$(basename "${file1/_1.fq/_merged.fq}")"
        "$USEARCH_PATH" -fastq_mergepairs "$file1" -reverse "$file2" -fastqout "$output_file" -fastq_nostagger
        echo "Merged: $file1 and $file2 -> $output_file"
    else
        echo "Warning: Corresponding file for $file1 not found!"
    fi
done
