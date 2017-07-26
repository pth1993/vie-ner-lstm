# Vietnamese Named Entity Recognition (vie-ner-lstm)
-----------------------------------------------------------------
Code by **Thai-Hoang Pham** at Alt Inc. 

## 1. Introduction
**vie-ner-lstm** is a fast-implementation of the system described in a paper [The Importance of Automatic Syntactic 
Features in Vietnamese Named Entity Recognition](https://arxiv.org/abs/1705.10610).
This system is used to recognizing named entities in Vietnamese texts and written by 
Python 2.7. The architecture of this system is two bidirectional LSTM layers followed by a feed-forward neural network. 
Finally, the output sequence is predicted by a softmax function. 

![](https://raw.githubusercontent.com/pth1993/vie-ner-lstm/master/docs/our_model.png "The architecture of 
vie-ner-lstm")

Our system achieved an F1 score of 92.05% on VLSP standard testset. The performance of our system with each feature set is 
described in the following table. 

| Word2vec | POS | Chunk | Regex |   F1   |
|:--------:|:---:|:-----:|:-----:|:------:|
|          |     |       |       | 62.87% |
|     x    |     |       |       | 74.02% |
|     x    |  x  |       |       | 85.90% |
|     x    |     |   x   |       | 86.79% |
|     x    |     |       |   x   | 74.13% |
|     x    |  x  |   x   |   x   | 92.05% |

## 2. Installation

This software depends on NumPy, Keras. You must have them installed before using **vie-ner-lstm**.

The simple way to install them is using pip:

```sh
	# pip install -U numpy keras
```
## 3. Usage

### 3.1. Data

The input data's format of vie-ner-lstm follows VLSP 2016 campaign format. There are four columns in this dataset 
including of **word**, **pos**, **chunk**, and **named entity**. For details, see sample data in **'data'** directory.
The table below describes an example Vietnamese sentence in VLSP dataset.

| Word      | POS | Chunk | NER   |
|-----------|-----|-------|-------|
| Từ        | E   | B-PP  | O     |
| Singapore | NNP | B-NP  | B-LOC |
| ,         | CH  | O     | O     |
| chỉ       | R   | O     | O     |
| khoảng    | N   | B-NP  | O     |
| vài       | L   | B-NP  | O     |
| chục      | M   | B-NP  | O     |
| phút      | Nu  | B-NP  | O     |
| ngồi      | V   | B-VP  | O     |
| phà       | N   | B-NP  | O     |
| là        | V   | B-VP  | O     |
| dến       | V   | B-VP  | O     |
| được      | R   | O     | O     |
| Batam     | NNP | B-NP  | B-LOC |
| .         | CH  | O     | O     |

To access the full dataset of VLSP 2016 campaign, you need to sign the user agreement of the VLSP consortium.
### 3.2. Command-line Usage

You can use vie-ner-lstm software by a following command:

```sh
	$ bash ner.sh
```

Arguments in ``ner.sh`` script:

* ``word_dir``:       path for word dictionary
* ``vector_dir``:         path for vector dictionary
* ``train_dir``:   path for training data
* ``dev_dir``:      path for development data
* ``test_dir``:      path for testing data
* ``num_lstm_layer``:      number of LSTM layers used in this system
* ``num_hidden_node``:     number of hidden nodes in a hidden LSTM layer
* ``dropout``:      dropout for input data (The float number between 0 and 1)
* ``batch_size``:      size of input batch for training this system.
* ``patience``:      number used for early stopping in training stage


**Note**: In the first time of running **vie-ner-lstm**, this system will automatically download word embeddings for 
Vietnamese from the internet. (It may take a long time because a size of this embedding set is about 1 GB). If the 
system cannot automatically download this embedding set, you can manually download it from here 
([vector](https://drive.google.com/open?id=0BytHkPDTyLo9WU93NEI1bGhmYmc), 
[word](https://drive.google.com/open?id=0BytHkPDTyLo9SC1mRXpkbWhfUDA)) and put it into **embedding** directory.

## 4. References

[Thai-Hoang Pham, Phuong Le-Hong, "The Importance of Automatic Syntactic Features in Vietnamese Named Entity 
Recognition" Proceedings of the 31th Pacific Asia Conference on Language, Information and Computation (PACLIC 31)](https://arxiv.org/abs/1705.10610)

## 5. Contact

**Thai-Hoang Pham** < phamthaihoang.hn@gmail.com >

Alt Inc, Hanoi, Vietnam
