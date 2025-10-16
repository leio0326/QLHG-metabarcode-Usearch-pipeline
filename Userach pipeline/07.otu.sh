#!/bin/bash
cd "your working path"
USEARCH_PATH="~/soft/usearch"
INPUT_DIR="02.Cutprimer"
OUTPUT_DIR="03.Merged"
mkdir -p "$OUTPUT_DIR"
usearch -sortbysize 06.Unique/all.uni.fa -fastaout 06.Unique/all.sort.fa -minsize 2
usearch -cluster_otus 06.Unique/all.sort.fa -otus 07.OTU/cluster/all.otus.fa -relabel OTU -uparseout 07.OTU/cluster/all_out -minsize 5
usearch -unoise3 06.Unique/all.sort.fa -zotus 07.OTU/unoise/zotus.fa -tabbedout 07.OTU/unoise/unoise3.txt
