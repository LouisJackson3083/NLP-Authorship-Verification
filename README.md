# NLP-Authorship-Verification
## Setup
Run these lines in your terminal and then in your IDE select the virtual environment to use with the notebook.
Windows:
```
py -m venv venv
venv\Scripts\activate.bat
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m pip install datasets transformers[sentencepiece]
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Linux/OSX:
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt  
python3 -m pip install transformers 
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


## Training
To run the training task run in the project directory 

``` shell
python3 -m src.train
```

To view training options run
``` shell
python3 -m src.train -h
```
