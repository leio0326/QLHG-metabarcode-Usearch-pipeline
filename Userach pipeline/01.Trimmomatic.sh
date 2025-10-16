#!/bin/bash
cd /data01/duanlei/ecology/rawdata/COI/20250115/00.Rawfq/
# 指定 Trimmomatic 的路径
TRIMMOMATIC_JAR="/public/home/wangwen_lab/zhoubotong/soft/trim/Trimmomatic-0.38/trimmomatic-0.38.jar"

# 创建输出文件夹（如果不存在）
OUTPUT_DIR="/data01/duanlei/ecology/rawdata/COI/20250115/01.Trimmomatic"

# 遍历当前目录中的所有 *_1.fq 文件
for file1 in *_1.fq; do
    # 确定对应的 *_2.fq 文件
    file2="${file1/_1.fq/_2.fq}"
    
    # 检查对应的 *_2.fq 文件是否存在
    if [[ -f "$file2" ]]; then
        # 使用 Trimmomatic 处理成对的 FASTQ 文件
        java -jar "$TRIMMOMATIC_JAR" PE -threads 20 "$file1" "$file2" \
            "$OUTPUT_DIR/${file1/_1.fq/_1_trimmed.fq}" "$OUTPUT_DIR/${file1/_1.fq/_1_unpaired.fq}" \
            "$OUTPUT_DIR/${file2/_2.fq/_2_trimmed.fq}" "$OUTPUT_DIR/${file2/_2.fq/_2_unpaired.fq}" \
            TRAILING:20 MINLEN:230
        echo "Processed: $file1 and $file2"
    else
        echo "Warning: Corresponding file for $file1 not found!"
    fi
done

