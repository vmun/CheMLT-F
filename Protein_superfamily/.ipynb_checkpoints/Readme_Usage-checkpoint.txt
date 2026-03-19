Usage:

1)Download original MMAT-DTA https://github.com/AronSchulman/MMAtt-DTA and include its data https://zenodo.org/records/10589696 inside the core folder of MMAT-DTA
2)Put convert_uniprot_to_sequence.py into MMAT-DTA and run it with python convert_uniprot_to_sequence.py to get SMILES
3)Take the resulting .csv files and do tokenizer preprocessing with MMAT-Prepare.ipynb (run it from the core CheMLT-F folder). You should get a new folder called inputddataMMAT which is the tokenized input
4)Run respective notebooks. If you have multiple GPUS available, the code makes it so that you can do parallel training for slt tasks if required