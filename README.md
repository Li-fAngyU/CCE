## Code for our work: Effective and Efficient Training for Sequential Recommendation Using Cumulative Cross-Entropy Loss

The following is a brief directory structure and description of this example:

```
├── data
    ├── data_process.py # the data process file.
    ├── generate_test.py # generate the negative example for BCE loss.
    ├── Beauty_item2attributes.json # Item attributes for S3-Rec pre-training task.
    ├── Beauty_sample.txt # Negative sampling for binary cross-entropy loss in advance.
    ├── Beauty.txt # The user interaction records after data process.
    ├── ...
├── final_result # the model training log folder.
    ├── GRU4Rec # unzip the GRU4Rec.zip file here.
    ├── S3Rec # unzip the S3Rec.zip file here.
    ├── SASRec # unzip the SASRec.zip file here.
    ├── RQ2_data # This data is downloaded by the tensorbord of the above files.
├── imgs
    ├── Intruduction.svg # image in paper introduction.
    ├── RQ1.png 
    ├── RQ2.svg
    ├── RQ3.svg
├── README.md
├── datasets.py
├── loss.py
├── models.py
├── modules.py
├── pre_exam.ipynb # generate the experiments result of RQ3.
├── run_finetune_full.py
├── run_pretrain.py  # for the pre-training task of S3-Rec.
├── SR_fig_process.ipynb # draw image file.
├── trainers.py
├── utils.py
```

Note that the GRU4Rec.zip, S3Rec.zip, and SASRec.zip files are the training log run by ourselves. And due to the limitation of file size, we upload separately.

## Environment
torch == 1.11.0
numpy==1.19.1
scipy==1.5.2
tqdm==4.48.2

python 3.7.3

os: windows/linux

## Reproduce
Here we give a example:
```python
python run_finetune_full.py --data_name LastFM --output_dir final_result/ --model_name GRU4Rec --istb 1 --loss_type CE_ALL
```
This example code means using CCE loss train GRU4Rec in LastFM datasets.

For data_name here are five choice: [Sports_and_Outdoors, Toys_and_Games, Yelp, Beauty, LastFM].

For loss_type [CE, PointwiseCE, CE_ALL], where CE reprensent the CE loss; PointwiseCE represent the BCE loss; CE_ALL represent the CCE loss.

### Therefore, the result in Table 2 and Fig 2 can reproduce by runing the code below: 
```python
#GRU4Rec
python run_finetune_full.py --data_name LastFM --output_dir final_result/ --model_name GRU4Rec --istb 1 --loss_type CE_ALL
python run_finetune_full.py --data_name LastFM --output_dir final_result/ --model_name GRU4Rec --istb 1 --loss_type CE
python run_finetune_full.py --data_name LastFM --output_dir final_result/ --model_name GRU4Rec --istb 1 --loss_type PointwiseCE
#To get the full result of GRU4Rec, need to change the data_name. This is similar to SASRec and S3Rec.

#SASRec
python run_finetune_full.py --data_name LastFM --output_dir final_result/ --model_name SASRec --istb 1 --loss_type CE_ALL
python run_finetune_full.py --data_name LastFM --output_dir final_result/ --model_name SASRec --istb 1 --loss_type CE
python run_finetune_full.py --data_name LastFM --output_dir final_result/ --model_name SASRec --istb 1 --loss_type PointwiseCE

#S3Rec
python run_finetune_full.py --data_name LastFM --output_dir final_result/ --model_name S3Rec --istb 1 --loss_type CE_ALL --ckp 150
python run_finetune_full.py --data_name Beauty --output_dir final_result/ --model_name S3Rec --istb 1 --loss_type CE_ALL --ckp 150
python run_finetune_full.py --data_name Sports_and_Outdoors --output_dir final_result/ --model_name S3Rec --istb 1 --loss_type CE_ALL --ckp 100
python run_finetune_full.py --data_name Toys_and_Games --output_dir final_result/ --model_name S3Rec --istb 1 --loss_type CE_ALL --ckp 150
python run_finetune_full.py --data_name Yelp --output_dir final_result/ --model_name S3Rec --istb 1 --loss_type CE_ALL --ckp 100
# Note that here has an additional parameter ckp, which depends on the pre-training model in reproduce folder.
# Besides, those pre-training models are offered by the source code of S3Rec.
```

Then, the training log is all saved in the final_result folder. And the experiment results in Table 2 and Fig 2 all came from the training log. 

### The experiment of Fig 3

Just run the code in pre_exam.ipynb, then will get the .csv file at final_result folder.

### Run SR_fig_process.ipynb file to get the images in Fig2 and 3.

## Our code is based on S3Rec's [source code](https://github.com/RUCAIBox/CIKM2020-S3Rec) to experiment. We are particularly grateful to it for providing the source code.

