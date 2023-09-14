# DeepSorption

This repository is the official implementation of **DeepSorption**. 


# Requirements
```
python         3.7
matplotlib     3.5.1
numpy          1.21.2
pandas         1.3.4
scikit-learn   0.22.1
torch          1.7.1
tqdm           4.42.1
mendeleev      0.9.0
```

# Dataset

We provide raw datasets and pre-processed datasets of material science under  https://doi.org/10.5281/zenodo.7699719.




# Demo


If you want to use our trained model directly for adsorption prediction, please excute `cd train_and_predict` and run: `python CoREMOF/pred.py`. 

The preprocessed training, validation and test set for Coremof dataset are stored in `save/coremof/COREMOF_train.npy`, `save/coremof/COREMOF_dev.npy`, and `save/coremof/COREMOF_test.npy`, respectively. This may take about 2 minutes to complete and report the performance. 


# Training example

To train DeepSorption on CoREMOF dataset, please excute `cd train_and_predict` and run: 

`python CoREMOF/main.py`


# Prediction example

To test the model performance on CoREMOF dataset, please excute `cd train_and_predict` and run:

`python CoREMOF/pred.py`


# EXPMOF example


To train and test DeepSorption on EXPMOF dataset, please excute `cd train_and_predict` and run: 

`python EXPMOF/leave_one_out.py --expmof 'c2h2'` or `python EXPMOF/leave_one_out.py --expmof 'co2'`

(Modify `--expmof` to specify different datasets.)



