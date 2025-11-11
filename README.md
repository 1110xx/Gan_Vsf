# Running the model

Datasets - METR-LA, SOLAR, ECG5000, ETTh1, ETTm1.

## Standard Training
```
python main.py --data ./data/{0} \
    --seed {1} \
    --model_name {2} --device cuda:0 \
    --expid {3} --epochs 100 --batch_size 64 --runs 2 \
    --random_node_idx_split_runs 10 \
    --step_size1 {4} \
    --w_fc 0.5 --w_imp 0.5
```
Here, <br />
{0} - refers to the dataset directory: ./data/{ECG000/METR-LA/SOLAR/ETTh1/ETTm1} <br />
{1} - refers to the random seed
{2} - refers to the forecasting backbone name <br />
{3} - refers to the manually assigned "ID" of the experiment  <br />
{4} - step_size1 is 2500 for METR-LA and SOLAR, 400 for ECG, 1000 for TRAFFIC, 1400 for ETTh1, 5800 for ETTm1



## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt


## Data Preparation

# Create data directories
mkdir -p data/{METR-LA,SOLAR,ECG5000, ETTh1, ETTm1}

# for any dataset, run the following command
python data_generation.py --ds_name {0} --output_dir data/{1} --dataset_filename data/{2}
```
Here <br />
{0} is for the dataset: metr-la, solar, ECG <br />
{1} is the directory where to save the train, valid, test splits. These are created from the first command <br />
{2} the raw data filename (the downloaded file), such as - ECG_data.csv, metr-la.hd5, solar.txt





