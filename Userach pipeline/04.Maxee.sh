#!/bin/bash
cd /data01/duanlei/ecology/rawdata/COI/20250115/

# 指定 usearch 的路径
USEARCH_PATH="/public/home/wangwen_lab/duanlei/soft/usearch"

# 创建输出文件夹（如果不存在）
OUTPUT_DIR_FILTERED="04.Maxee"
mkdir -p "$OUTPUT_DIR_FILTERED"

# 遍历合并后的 FASTQ 文件
for merged_file in 03.Merged/*_merged.fq; do
    # 生成输出过滤后的文件名，转换为 .fasta 格式
    filtered_file="$OUTPUT_DIR_FILTERED/$(basename "${merged_file/_merged.fq/.MAXEE_1.fasta}")"
    
    # 使用 usearch 进行过滤并转换为 FASTA 格式
    "$USEARCH_PATH" -fastq_filter "$merged_file" -fastaout "$filtered_file" --fastq_maxee 1
    
    echo "Filtered and converted: $merged_file -> $filtered_file"
done

