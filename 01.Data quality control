# 01.Data quality control
#!/bin/bash
cd "your working path"
#Trimmomatic path" 
TRIMMOMATIC_JAR="~/soft/trim/Trimmomatic-0.38/trimmomatic-0.38.jar"
#Creat output path
OUTPUT_DIR="01.Trimmomatic"
for file1 in *_1.fq; do
    file2="${file1/_1.fq/_2.fq}"
    if [[ -f "$file2" ]]; then

        java -jar "$TRIMMOMATIC_JAR" PE -threads 20 "$file1" "$file2" \
            "$OUTPUT_DIR/${file1/_1.fq/_1_trimmed.fq}" "$OUTPUT_DIR/${file1/_1.fq/_1_unpaired.fq}" \
            "$OUTPUT_DIR/${file2/_2.fq/_2_trimmed.fq}" "$OUTPUT_DIR/${file2/_2.fq/_2_unpaired.fq}" \
            TRAILING:20 MINLEN:230
        echo "Processed: $file1 and $file2"
    else
        echo "Warning: Corresponding file for $file1 not found!"
    fi
done
