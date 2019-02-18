import os
import argparse
import numpy                      as np
from tensorflow.keras.utils       import to_categorical
from tensorflow.keras.models      import Sequential
from tensorflow.keras.layers      import Dense, LSTM, Dropout
from tensorflow.keras.layers      import Embedding
from tensorflow.keras.layers      import BatchNormalization
from sklearn.pipeline             import Pipeline, FeatureUnion
from sklearn.externals            import joblib
from sklearn.model_selection      import train_test_split
from dependencies                 import tokenize_document, DocTokenizer, WordsEncoder, Padder, load_dataset, load_glove_embeddings
from polyaxon_client.tracking     import Experiment, get_log_level, get_data_paths, get_outputs_path
from utils                        import gcs_upload_blob, gcs_bucket_prefix

def create_preprocessing_pipeline(top_words=20000, max_sequence_length=500):

    preprocessing_pipeline = Pipeline([
        ('tokenizer', DocTokenizer()),
        ('encoder', WordsEncoder(top_words=top_words)),
        ('padder', Padder(max_sequence_length=max_sequence_length))])

    return preprocessing_pipeline

def create_embedding_matrix(glove_embeddings, preprocessing_pipeline):

    word_index = preprocessing_pipeline.named_steps['encoder'].encoder_.word_index
    embedding_dim = len(glove_embeddings['yes']) #=300
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = glove_embeddings.get(word)

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def create_model(embedding_matrix, trainable_embedding=True):

    model = Sequential()
    model.add(Embedding(np.shape(embedding_matrix)[0], np.shape(embedding_matrix)[1], weights=[embedding_matrix], trainable=trainable_embedding))
    model.add(LSTM(100))
    model.add(Dense(70, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(BatchNormalization())
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--top_words', default=20000, type=int)
    parser.add_argument('--max_sequence_length', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    args                = parser.parse_args()
    arguments           = args.__dict__
    num_epochs          = arguments.pop('num_epochs')
    top_words           = arguments.pop('top_words')
    max_sequence_length = arguments.pop('max_sequence_length')
    batch_size          = arguments.pop('batch_size')

    try:
        # if running in polyaxon
        experiment = Experiment()
        data_path = get_data_paths()['data-local']
    except:
        # if running locally
        data_path = '/data'

    np.random.seed(7)
    bbc_data_dir           = data_path + "/bbc-topic-classification/bbc_data/"
    glove_embedding_dir    = data_path + "/bbc-topic-classification/glove.6B.300d.txt"

    data                   = load_dataset(bbc_data_dir)
    glove_embeddings       = load_glove_embeddings(glove_embedding_dir)

    preprocessing_pipeline = create_preprocessing_pipeline(top_words, max_sequence_length)

    train, test            = train_test_split(data, test_size=0.25)
    X_train                = preprocessing_pipeline.fit_transform(train.text)
    y_train                = to_categorical(train["class"])

    embedding_matrix                   = create_embedding_matrix(glove_embeddings, preprocessing_pipeline)
    model                              = create_model(embedding_matrix)
    model.fit(X_train, y_train, epochs = num_epochs, batch_size=batch_size, shuffle=True)

    model.save('model')
    joblib.dump(preprocessing_pipeline, 'preprocessing_pipeline.pkl')

    X_test  = preprocessing_pipeline.transform(test.text)
    y_test  = to_categorical(test["class"])
    metrics = model.evaluate(X_test, y_test)

    try:
        # if running in polyaxon
        bucket, prefix                          = gcs_bucket_prefix(get_outputs_path())
        destination_model_path                  = '{}/{}'.format(prefix, 'model')
        gcs_upload_blob(bucket, 'model', destination_model_path)

        destination_preprocessing_pipeline_path = '{}/{}'.format(prefix, 'preprocessing_pipeline.pkl')
        gcs_upload_blob(bucket, 'preprocessing_pipeline.pkl', destination_preprocessing_pipeline_path)

        experiment.log_metrics(
            loss     = metrics[0],
            accuracy = metrics[1]
        )
    except:
        # if running locally
        print("loss: {}, accuracy: {}".format(metrics[0], metrics[1]))
