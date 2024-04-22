COMP34812_modelcard_template.md

{{ card_data }}

---

# Model Card for {{ model_id | default("My Model", true) }}

<!-- Provide a quick summary of what the model is/does. -->

{{ model_summary | default("", true) }}


## Model Details

### Model Description

This model attempts to learn to identify if pairs of texts are written by the same author.
It learns the relationship between texts written by the same author, from vectors of encoded text, vectors of punctuation, and vectors of part of speech. It collates these encoded vectors together and makes a class prediction using the T5 Large transformer model.

- **Developed by:** Aaron-Teodor Panaitescu , Louis Jackson 
- **Language(s):** Python
- **Model type:** Language Model
- **Model architecture:** Transformer
- **Finetuned from model:** google-t5/t5-large

### Model Resources

- **Repository:** [Text-to-Text Transfer Transformer](https://github.com/google-research/text-to-text-transfer-transformer)
- **Paper or documentation:** [Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer](https://jmlr.org/papers/volume21/20-074/20-074.pdf)

## Training Details

### Training Data

- Train - 30,000 items of text pairs & labels.
- Dev - 6,000 items of text pairs & labels.
- Test - ?? items of text pairs.

### Training Procedure

The model 

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

{{ hyperparameters | default("[More Information Needed]", true)}}

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

## Technical Specifications

### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

### Software

{{ software | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

{{ bias_risks_limitations | default("[More Information Needed]", true)}}

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

{{ additional_information | default("[More Information Needed]", true)}}

