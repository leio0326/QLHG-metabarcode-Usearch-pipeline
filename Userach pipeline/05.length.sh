#!/bin/bash

# 创建输出目录（如果不存在）
mkdir -p 05.length

# 遍历 04.Maxee 目录中的所有 .MAXEE_1.fasta 文件
for fasta_file in 04.Maxee/*.MAXEE_1.fasta; do
    # 提取文件名（去掉路径和扩展名）
    base_name=$(basename "$fasta_file" .MAXEE_1.fasta)
    
    # 生成输出文件名
    output_file="05.length/${base_name}.fa"
    
    # 执行 seqkit 命令
    seqkit seq -m 286 -M 322 "$fasta_file" > "$output_file"
    
    echo "Processed: $fasta_file -> $output_file"
done

