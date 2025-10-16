#!/bin/bash
cd /data01/duanlei/ecology/rawdata/COI/20250115/

# 指定 usearch 的路径
USEARCH_PATH="/public/home/wangwen_lab/duanlei/soft/usearch"

# 创建输出文件夹（如果不存在）
OUTPUT_DIR="03.Merged"
mkdir -p "$OUTPUT_DIR"

# 遍历当前目录中所有的 *_1.fq 文件
for file1 in 02.Cutprimer/*_1.fq; do
    # 确定对应的 *_2.fq 文件
    file2="${file1/_1.fq/_2.fq}"
    
    # 检查对应的 *_2.fq 文件是否存在
    if [[ -f "$file2" ]]; then
        # 生成输出文件名，去掉文件后缀
        output_file="${OUTPUT_DIR}/$(basename "${file1/_1.fq/_merged.fq}")"

        # 使用 usearch 合并成对的 FASTQ 文件
        "$USEARCH_PATH" -fastq_mergepairs "$file1" -reverse "$file2" -fastqout "$output_file" -fastq_nostagger
        
        echo "Merged: $file1 and $file2 -> $output_file"
    else
        echo "Warning: Corresponding file for $file1 not found!"
    fi
done

