# COMP90051_Project2

`code/`: main code to implement the project
Steps to experiments
1. Run `split_data.ipynb` to split the train and evaluation data. The embeeding methods are in `preprocessing.py`, `metapath2vec` and `doc2vec.ipynb`.
2. Run `mll-abstract-title.ipynb`, `mll-authors.ipynb` and `mll-year-venue.ipynb` to train the data and get prediction on validation data. The models are in `NN_Models.py`.
3. Run `test.ipynb` to stack the results from three models and get the best weights and threshold.
4. Run `Kaggle.ipynb` to predict the test data, and save the csv file in `kaggle/` root.

