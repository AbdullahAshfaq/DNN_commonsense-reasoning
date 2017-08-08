from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import pandas as pd
import numpy as np

# Set paths to the model.
VOCAB_FILE = "/home/ab/PycharmProjects/SNLIAssociation/skip_thoughts_bi_2017_02_16/vocab.txt"
EMBEDDING_MATRIX_FILE = "/home/ab/PycharmProjects/SNLIAssociation/skip_thoughts_bi_2017_02_16/embeddings.npy"
CHECKPOINT_PATH = "/home/ab/PycharmProjects/SNLIAssociation/skip_thoughts_bi_2017_02_16/model.ckpt-500008"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
#MR_DATA_DIR = "/dir/containing/mr/data"

# Set up the encoder. Here we are using a single unidirectional model.
# To use a bidirectional model as well, call load_model() again with
# configuration.model_config(bidirectional_encoder=True) and paths to the
# bidirectional model's files. The encoder will use the concatenation of
# all loaded models.
encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(bidirectional_encoder=True),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)


X = pd. read_csv('train.csv')


encodings_s1 = encoder.encode(X.iloc[0:100000,2])
encodings_s2 = encoder.encode(X.iloc[0:100000,3])

encodings_s1 = pd.DataFrame(encodings_s1)
encodings_s2 = pd.DataFrame(encodings_s2)

join = pd.concat([encodings_s1,encodings_s2],axis=1)
join.to_csv('train_vec1.csv',sep=",")
print(join.shape)

#encodings_s1.to_csv('enc_s1.csv',sep=",")
#encodings_s2.to_csv('enc_s2.csv',sep=",")
print('Success1')

encodings_s2 = encoder.encode(X.iloc[200000:300000,3])
encodings_s1 = encoder.encode(X.iloc[200000:300000,2])

encodings_s1 = pd.DataFrame(encodings_s1)
encodings_s2 = pd.DataFrame(encodings_s2)

join = pd.concat([encodings_s1,encodings_s2],axis=1)
join.to_csv('train_vec3.csv',sep=",")
print(join.shape)
print('Success3')


encodings_s1 = encoder.encode(X.iloc[300000:400000,2])
encodings_s2 = encoder.encode(X.iloc[300000:400000,3])

encodings_s1 = pd.DataFrame(encodings_s1)
encodings_s2 = pd.DataFrame(encodings_s2)

join = pd.concat([encodings_s1,encodings_s2],axis=1)
join.to_csv('train_vec4.csv',sep=",")
print(join.shape)
print('Success4')


encodings_s1 = encoder.encode(X.iloc[400000:500000,2])
encodings_s2 = encoder.encode(X.iloc[400000:500000,3])

encodings_s1 = pd.DataFrame(encodings_s1)
encodings_s2 = pd.DataFrame(encodings_s2)

join = pd.concat([encodings_s1,encodings_s2],axis=1)
join.to_csv('train_vec5.csv',sep=",")
print(join.shape)
print('Success5')


encodings_s1 = encoder.encode(X.iloc[500000:,2])
encodings_s2 = encoder.encode(X.iloc[500000:,3])

encodings_s1 = pd.DataFrame(encodings_s1)
encodings_s2 = pd.DataFrame(encodings_s2)

join = pd.concat([encodings_s1,encodings_s2],axis=1)
join.to_csv('train_vec6.csv',sep=",")
print(join.shape)
print('Success6')



