#!/bin/bash
export JT_SYNC=1

now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal.yaml
labeled_id_path=partitions/pascal/92/labeled.txt
unlabeled_id_path=partitions/pascal/92/unlabeled.txt
save_path=exp/pascal/92/corrmatch


mkdir -p $save_path


python -m corrmatch --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path --save-path $save_path 2>&1 | tee $save_path/${now}.txt
