# DrugANNs
Global AI Challenge solution
Overall pipeline:
* data preprocessing (removed unneeded parts of molecules)
* generated Morgan, MACCS and Estate fingerprints
* applied MolCLR graph neurla network
* applied RandomForest to the features described before
* the models' results were merged and averaged
* the results from the previous point were also passed to the Lipinski rule checker
# Repository structure
* notebooks - contains all the notebooks which were used during the analysis
* data - folder with all the data we used
* MolCLR - directory with MolCLR model
* YouGraphRF - directory with random forest model
# Model running
1. Run `data_preprocessing.ipynb` to make canonical SMILES
2. Run `ogb-rdk-transform.ipynb` to get preprocessed dataset
3. Go to `YouGraphRF` and run `python random_forest.py --smiles_file ... --smiles_test_file ...`
4. Take predictions from `rf_preds/rf_final_pred.npy`
5. Go to MolCLR
6. Place preprocessed molecules data to `data/covid/COVID.csv` and `data/covid/COVID-test.csv` for train and test subsets correspondingly.
7. Run `python finetune_contrast.py`
8. Finally, run `predict-molclr.ipynb`. You need to change model path with your checkpoint. Or you can find checkpoint used for submission in finetune folder
9. The final predictions should be passed to `lipinski_rule_application.ipynb`
# Requirements
You can find the requirements in requirements.txt file 

