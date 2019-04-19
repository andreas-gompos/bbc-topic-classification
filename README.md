# Topic Classifier

## Description

In this project a topic classification model was created. More specifically an LSTM network was trained to distinguish between 5 different categories of articles (business, entertainment, politics, sport, tech). The dataset used, for training the network, was the [BBC articles dataset](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip), which consists of 2225 documents, from the BBC news website corresponding to stories from 2004-2005. The model is deployed on Kubernetes on GKE and can used at [datagusto.com](https://datagusto.com).

## Train the Model

```bash
#!/bin/bash
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt wordnet
python ./src/train.py --num_epochs={{ num_epochs }}
                      --top_words={{ top_words }}
                      --max_sequence_length={{ max_sequence_length }}
                      --batch_size={{ batch_size }}
                      --polyaxon_env={{ polyaxon_env }}
```

| Parameter               |  Description                                                                            | Valid Values | Default   |
| ---                     | ---                                                                                     | ---          | ---       |
| num_epochs              | Number of epochs to be used for training                                                | int          | 40        |
| batch_size              | Batch size to be used for training                                                      | int          | 256       |
| top_words               | Number of most common words to be used for training (rare words will be dropped)        | int          | 35000     |
| max_sequence_length     | Fixed length of each input text. The text will be padded or trimmed down to that length | int          | 500       |
| polyaxon_env            | Indicate if running in Polyaxon                                                         | 0, 1         | 0         |

Any feedback is welcome! :)
[LinkedIn](https://www.linkedin.com/in/andreas-gompos/)
