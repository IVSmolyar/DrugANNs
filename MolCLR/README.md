Based on https://github.com/yuyangw/MolCLR


1. Place preprocessed molecules data to `data/covid/COVID.csv` and `data/covid/COVID-test.csv` for train and test subsets correspondingly.
2. `python finetune_contrast.py`
3. Finally, run `predict-molclr.ipynb`. You need to change model path with your checkpoint. Or you can find checkpoint used for submission in finetune folder

