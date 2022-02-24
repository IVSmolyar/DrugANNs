Based on https://github.com/PierreHao/YouGraph

1. Convert smiles to dicts using OGB smiles2graph with `ogb-rdk-transform.ipynb`
2. `python extract_fingerprint.py --processed_mol_path ... --train_data ... --test_data ...`
3. `python random_forest.py --smiles_file ... --smiles_test_file ...`
4. Take predictions from `rf_preds/rf_final_pred.npy`

Also we provide `rf_preds/rf_final_pred.npy` that was used in our submission