from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import glob
import re


class NlpPredict():

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def set_sentence(self, sentence):
        # Training data CSV read using pandas
        training_path = "train/"
        training_csv_files = glob.glob(training_path + "/*.csv")
        training_df_list = (pd.read_csv(file,
                                        names=["sentence", "label"],
                                        skipinitialspace=True,
                                        skiprows=1,
                                        engine="python") for file in training_csv_files)
        training_big_df = pd.concat(training_df_list, ignore_index=True)
        training_big_df = training_big_df.drop(training_big_df[training_big_df.label == 2].index)
        sentiment_train = training_big_df

        # Convert training sentences from pandas dataframe to list
        training_sentences_df_list = training_big_df.loc[:, 'sentence']
        training_sentences = training_sentences_df_list.values.tolist()

        # Convert training labelss from pandas dataframe to list
        training_labels_list = training_big_df.loc[:, 'label']
        training_labels = training_labels_list.values.tolist()

        training_size = len(sentiment_train)

        # Split the sentences
        training_sentences_split = training_sentences[:training_size]

        training_sentences = []

        for i in training_sentences_split:
            url_remove = re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', '', i)
            url_remove = re.sub(r'http\S+', '', url_remove)
            url_remove = re.sub(r'@+', '', url_remove)
            url_remove = url_remove.lower().rstrip()
            cleaned = url_remove
            training_sentences.append(cleaned)

        training_length = len(training_sentences)

        # Split the labels
        training_labels_split = training_labels[:training_size]

        training_labels = training_labels_split

        # Initialize the Tokenizer class
        tokenizer = Tokenizer(oov_token="<OOV>")

        # Generate the word index dictionary
        tokenizer.fit_on_texts(training_sentences)

        # Print the length of the word index
        word_index = tokenizer.word_index

        # Generate and pad the sequences
        sequences = tokenizer.texts_to_sequences(training_sentences)
        padded = pad_sequences(sequences, maxlen=100, padding='post')

        # Print a sample sentence
        index = 2

        vocab_size = max([len(tokenizer.word_index)]) + 1
        max_length = max([len(i.split()) for i in training_sentences])
        trunc_type = 'post'
        padding_type = 'post'
        oov_tok = "<OOV>"

        # Initialize the Tokenizer class
        tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

        # Generate the word index dictionary
        tokenizer.fit_on_texts(training_sentences)
        word_index = tokenizer.word_index

        self.padded = pad_sequences(tokenizer.texts_to_sequences([sentence]), maxlen=91)

    def predict(self):
        res_predict = self.model.predict(self.padded)
        res_np_float = res_predict.astype(np.float)
        self.res = float(res_np_float[0][0])

        return self.res

    def sumRatingModel(self, rating, lebel, nlp_score):
        if lebel == "Positif":
            if 3 <= rating < 5:
                result = float(rating) + nlp_score
                return round(result, 2)
            elif rating == 5:
                result = 5
                return result
        elif lebel == "Negatif":
            if 3 > rating > 0:
                result = float(rating) - nlp_score
                return round(result, 2)


# if __name__ == '__main__':
#     x = NlpPredict('nlp_model/sentimentanalysisv4.h5')
#     x.set_sentence("Orang ini baik")

#     r = x.predict()
#     print(r)
#     print(type(r))