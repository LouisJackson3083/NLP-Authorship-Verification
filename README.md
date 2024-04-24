# NLP-Authorship-Verification
This folder has the following twio notebooks that can easily run the training/inference script. They are desigend to be ran in colab so they will clone the repository where this is hosted code is hosted, that way they will have access to the coursework data directly. To use different data files add flags `--data-dir`, `--data-[stage]` where stage can be `train` `test` `dev`. 

For example, if you want to have train data in `./new_data/new_train.csv` you would pass `--data-dir=new_data --data-train=new_train.csv` to python3 src/train.py

## Setup
In a terminal run these these lines.
```
pip install -r requirements.txt
```

## Training
To run the training task run was run in the project directory with the following flags.
``` shell
python3 ./src/train.py --lr 0.0001 --batch_size 64 --num_workers 32 --dropout 0.5 --num_epochs 100 --ratio 0.05 --patience 15
```

To view other training options run
``` shell
python3 ./src/train -h
``` 

## Inference
To do inference, you will need a model. The inference script also needs a model checkpoint to use. This is expected to be in `assets/saved/AV/best/best.ckpt`. This h=is being hosted on GoogleDrive [here](https://drive.google.com/file/d/1IX5Gj60QI0y5zn3J1Pt3Dkv-Yu8GWjjL/view?usp=sharing). Download the model and put it into the correct folder & name.

To run the inference script just run
``` shell
python3 ./src/inference
``` 

## Using Colab

The [following](https://colab.research.google.com/drive/1dhbCc52-VFv8ZQ49u5gH5Kb7PwvPE5sB#scrollTo=df3w99OWh6os) notebook is hosted on colab. It clones the repo of this project which also contains the training and testing data. It then installs the requirements. The training script can simply be ran in this notebook. To do inference just add the model as explained in the previous part and run the inference script in the notebook.
