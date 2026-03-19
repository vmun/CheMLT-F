Use:

1)Download original BALM folder from https://github.com/meyresearch/BALM/tree/main
2)Add contents from this folder to the downloaded folder
3)Download pretrained model checkpoint (https://huggingface.co/BALM/bdb-cleaned-r-esm-lokr-chemberta-loha-cosinemse) and BindingDB data (https://huggingface.co/datasets/BALM/BALM-benchmark) to folder
4)Install requirements and environment
5)Run commands below by specifying train/test csvs of KIBA and Davis. Either copy them from CheMLT-F in to a new data folder inside BALM, or update the arguments train_csv and test_csv to the actual path of KIBA and Davis


python finetune_and_evaluate.py --config_filepath configs/bindingdb_random/esm_lokr_chemberta_loha_cosinemse_1.yaml --train_csv data/davis_train.csv --test_csv data/davis_test.csv --dataset davis --output_csv results/davis_finetuned.csv --device cuda:0 --epochs 100 --bindingdb_csv BindingDB_filtered/data.csv --save_model results/davis_checkpoint.bin --batch_size 64

python finetune_and_evaluate.py --config_filepath configs/bindingdb_random/esm_lokr_chemberta_loha_cosinemse_1.yaml --train_csv data/kiba_train.csv --test_csv data/kiba_test.csv --dataset kiba --output_csv results/kiba_finetuned.csv --device cuda:0 --epochs 100 --bindingdb_csv BindingDB_filtered/data.csv --save_model results/kiba_finetuned_model.bin --batch_size 64