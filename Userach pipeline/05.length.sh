# Control the sequence length within the target primer length range
#!/bin/bash
cd "your working path"
INPUT_DIR="04.Maxee"
OUTPUT_DIR="05.length"
mkdir -p "$OUTPUT_DIR"
for fasta_file in "$INPUT_DIR"/*.MAXEE_1.fasta; do
    base_name=$(basename "$fasta_file" .MAXEE_1.fasta)
    output_file="05.length/${base_name}.fa"
    # Here, the range of primer target length is set, where - m represents the minimum value and - M represents the maximum value
    seqkit seq -m 286 -M 322 "$fasta_file" > "$output_file"   
    echo "Processed: $fasta_file -> $output_file"
done

