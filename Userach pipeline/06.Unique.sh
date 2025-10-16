#!/bin/bash
cd /data01/duanlei/ecology/rawdata/COI/20250115/05.length

for input_file1 in *.fa; do
    
    file_prefix="${input_file1%.fa}"
     
    usearch -fastx_uniques $input_file1 -fastaout ../06.Unique/${file_prefix}.uni.fa -sizeout 

done

