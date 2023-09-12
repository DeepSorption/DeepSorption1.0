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

We provide raw datasets of material science under `raw_data/`.

For convenience, we also provide pre-processed datasets for download: 

- For CoREMOF dataset, download from [CoREMOF](https://drive.google.com/drive/folders/1GOCK6z7c1Vn9-HCUAlsXtpis0aKwmnXS?usp=sharing) (203.7MB), put it under `pre_data/CoREMOF`.

- For hMOF dataset, download from [hMOF](https://drive.google.com/drive/folders/1GvxCq-Us0QrQ9Cpyut799zSkQNBBYvuS?usp=sharing) (1.82GB), put it under `pre_data/hMOF`.

- For EXPMOF dataset, pre-processed data can be generated by running `train_and_predict/EXPMOF/dataloader_c2h2.py` and `train_and_predict/EXPMOF/dataloader_co2.py`. (Don't forget to unrar `car.rar` under `raw_data/EXPMOF/C2H2` and `raw_data/EXPMOF/CO2` before pre-process.)


# Demo

For your convenience, we provide the trained model, stored in `save/coremof/CoREMOF_model.pt`.

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



