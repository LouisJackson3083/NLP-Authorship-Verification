COMP34812_modelcard_av_t5.md
---

# Model Card for Model 1

This model does authorship verification: given two text samples it predicts whether they are or not from the same author.
It extracts POS and punctuation data from the texts, feeds themm through a transformer encoder, combines POS and text information in an attention block, applyies a CNN on the punctuation data and combines the output to generate a prediction. 
<!-- Provide a quick summary of what the model is/does. -->

## Model Details

### Model Description

This model attempts to learn to identify if pairs of texts are written by the same author.
It learns the relationship between texts written by the same author, from vectors of encoded text, vectors of punctuation, and vectors of part of speech. It collates these encoded vectors together and makes a class prediction using the T5 Large transformer model.

- **Developed by:** Aaron-Teodor Panaitescu , Louis Jackson 
- **Language(s):** Python
- **Model type:** Language Model
- **Model architecture:** Transformer Encoder Based 
```text
              / Texts       -> Tokenizer -> T5 Encoder --\
 Input Texts -- POS         -> Indexer   -> T5 Encoder --> Attention    *-> Classifier
              \ Punctuation -> Tokenizer -> T5 Encoder --> Convolution -/
```
- **Finetuned from model:** google-t5/t5-large 

### Model Resources

- **Repository:** [Text-to-Text Transfer Transformer](https://github.com/LouisJackson3083/NLP-Authorship-Verification/) 
- **Paper used:** [Text-to-Text Transformer in Authorship Verification Via Stylistic and Semantical Analysis](https://ceur-ws.org/Vol-3180/paper-215.pdf)

### Training Data

- Train - 27,000 items of text pairs & labels.
- Validation - 3,000 items of text pairs & lables.

Preprocessing:
 - Text: Text is passed as it is in the dataset 
 - Part of Speech: extract using [`nltk.tag.pos_tag`](https://www.nltk.org/api/nltk.tag.pos_tag.html), then index 
 - Punctuation: extract punctuation (using custom regex), join in space spearated string.
Tokenization: 
 - Text and punctuation all tokenezied separately using [T5Tokenizer](https://huggingface.co/docs/transformers/en/model_doc/t5) 

### Training Procedure

Model is fitted using the AdamOptimizer over several epochs on the training data.

Encoding:
 - [T5EncoderModel](https://huggingface.co/docs/transformers/en/model_doc/t5) is used to encode each embedding 

#### Training Hyperparameters


- `save_top_k: int = 1` Number of top model checkpoints to save based on a certain metric, typically validation accuracy. 
- `num_workers: int = 10` Number of worker processes used in the training process.
- `num_epochs: int = 7` Number of training cycles through the entire dataset.
- `batch_size: int = 8` Number of samples processed in parallel before the model's weights are updated.
- `max_len: int = 350` Maximum length of the input sequences.
- `lr: float = 2e-5` The learning rate of the optimizer. 
- `num_filters: int = 128` Number of output filters for the punctuation CNN.
- `filter_sizes: list[int] = [1,2,3]` Size of the convolutional filters. For example, sizes 1, 2, and 3 can capture 1-gram, 2-gram, and 3-gram features respectively.
- `dropout: float = 0.15` Dropout rate during training.
- `embedding_dim: float = 128` Size of the embedding vectors. All words or tokens are converted into vectors of this size before being processed by the neural network.
- `ratio: float = 1.0` Procentage of training saples to use
- `patience: int = 7` Number of epochs to continue training without improvement in the validation accuracy before stopping the training early. 

<!--  {{ hyperparameters | default("[More Information Needed]", true)}} -->
### Speeds, Sizes, Times
! python ./src/train.py --lr 0.0001 --batch_size 64 --num_workers 64 --dropout 0.5 --num_epochs 100 --patience 15 --ratio 0.5

The model was trained with the following parameters
| eta | batch size | workers | dropout | epochs | patience | ratio |
| ----|------------|---------|---------|--------|----------|------ | 
| 1e-4| 64 | 64 | 0.5 | 100 | 15 | 0.5 | 

This resulted in 18 epochs taking with a time of ~4.2 min/epoch.
<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


## Evaluation
The best performing model acheived:
- Training Accuracy: 0.5404
- Training Loss: 0.5950 
- Validation Accurcy: 0.5304
- Validation Loss: 0.6103 

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data
- Test - 6,000 items of text pairs. 

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->


#### Metrics
The metrics used will be acurracy and loss.

<!-- These are the evaluation metrics being used. -->


### Results
Testing labels are not available, so we can't compute testing metrics yet.


## Technical Specifications

### Hardware
Used: L4 TensorCore GPU on Google Colab 
Neeeded: 15GiB RAM

### Software
Python3, CUDA

**Python3 Dependencies**
```text
ipykernel
tqdm
transformers
sentencepiece
torch
pandas
nltk
scikit-learn
emoji
pytorch-lightning
```


## Additional Information 

<!-- Any other information that would be useful for other people to know. -->

 
