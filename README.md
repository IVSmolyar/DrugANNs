# DrugANNs
Global AI Challenge solution
Overall pipeline:
* data preprocessing (removed )
* generated Morgan, MACCS and Estate fingerprints
* applied MolCLR graph neurla network
* applied RandomForest to the features described before
* the models' results were merged and averaged
* the results from the previous point were also passed to the Lipinski rule checker
# Repository structure
* notebooks - contains all the notebooks which were used during the analysis
* data - folder with all the data we used
* scripts - folder with the scripts and models which we used