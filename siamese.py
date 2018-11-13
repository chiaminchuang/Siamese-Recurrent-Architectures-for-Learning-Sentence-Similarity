import csv
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
max_len = 100
import tensorflow as tf

from SiameseLSTM import SiameseLSTM

config = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'emb_dim': 50,
    'hidden_size': 50,
    'max_len': 100,
    'num_epoch': 360,
    'max_grad_norm': 5,
    'vocab_size': 400001
}

def load_dataset():
    def preprocess(sentence):
        return [lemmatizer.lemmatize(word) for word in word_tokenize(sentence.lower())]        
    def padding_mask(sentence):
        mask = np.zeros((max_len))
        mask[len(sentence) - 1] = 1
        for i in range(0, max_len - len(sentence)):
            sentence.append('<unknown>')
        
        return np.array(sentence), mask
        
        
    filepath = 'data/sick.csv'
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        sentences_A, sentences_B, relatedness, masks_A, masks_B =  [], [], [], [], []
        for row in reader:
            sentence_A, mask_A = padding_mask(preprocess(row[1]))
            sentences_A.append(sentence_A)
            masks_A.append(mask_A)
            
            sentence_B, mask_B = padding_mask(preprocess(row[2]))
            sentences_B.append(sentence_B)
            masks_B.append(mask_B)
            
            relatedness.append(float(row[3]))
    
    return np.array(sentences_A), np.array(sentences_B), np.array(relatedness), np.array(masks_A), np.array(masks_B)

def load_wordembedding(filepath='wordvector/glove.6B.50d.txt', emb_dim=50):
    
    embeddings = [[.0]*emb_dim]
    word2id = {'<unknown>': 0} 
    id2word = {0: '<unknown>'}
    
    
    with open(filepath, 'r', encoding='utf-8') as file:
        for i, embedding in enumerate(file):
            embedding = embedding.rstrip('\n').split(' ')
            embeddings.append([float(emb) for emb in embedding[1:]])
            word2id[embedding[0]] = int(i+1)
            id2word[str(i+1)] = embedding[0]
    
#     embeddings.append(np.zeros(emb_dim, dtype=float))
#     word2id['<unknown>'] = len(embeddings)
#     id2word[str(len(embeddings))] = '<unknown>'
    return np.array(embeddings), word2id, id2word

def word_to_id(sentences, word2id):
    
    length = len(sentences)
    for i in range(length):
        for j, word in enumerate(sentences[i]):
            if not word in word2id:
                word = '<unknown>'
            sentences[i][j] = word2id[word]
            
    return sentences

def id_to_word(sentences, id2word):
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            sentences[i][j] = id2word[sentences[i][j]]
    return sentences


import math

def next_batch(data, batch_size):
    
    
    sentences_A, sentences_B, y_train, masks_A, masks_B = data
    size = len(sentences_A)
    indexes = np.arange(0, size)
    np.random.shuffle(indexes)
    
    sentences_A = [sentences_A[i] for i in indexes]
    sentences_B = [sentences_B[i] for i in indexes]
    masks_A = [masks_A[i] for i in indexes]
    masks_B = [masks_B[i] for i in indexes]
    y_train = [y_train[i] for i in indexes]
    
    
    nbatch = math.ceil(size / batch_size)
    for i in range(nbatch):
        offset = i * batch_size
        sentence_A = sentences_A[offset: offset + batch_size]
        sentence_B = sentences_B[offset: offset + batch_size]
        mask_A = masks_A[offset: offset + batch_size]
        mask_B = masks_B[offset: offset + batch_size]
        relatedness = y_train[offset: offset + batch_size]
        
        yield [sentence_A, sentence_B, relatedness, mask_A, mask_B]

    
    
def run_epoch(model, session, data, global_steps):
    
    for step, (sentence_A, sentence_B, relatedness, mask_A, mask_B) in enumerate(next_batch(data, config['batch_size'])):
        feed_dict = {
            model.sentence_A: sentence_A,
            model.sentence_B: sentence_B,
            model.mask_A: mask_A,
            model.mask_B: mask_B,
            model.relatedness: relatedness
        }
        
        fetches = [model.loss, model.similarity, model.train_op]
        loss, similarity, _ = session.run(fetches, feed_dict)
        
        if global_steps % 100 == 0:
            print('Step {}, Loss: {}'.format(global_steps, loss))
        
        global_steps += 1
        
    return global_steps
        
        

def train():
    with tf.Session() as session:
        
        print('Loading dataset ...')
        sentences_A, sentences_B, relatedness, masks_A, masks_B = load_dataset()
        print('Loading dataset completed')
        
        print('Loading word embedding ...')
        embeddings, word2id, id2word = load_wordembedding(emb_dim=config['emb_dim'])
        print('Loading word embedding completed')
        
        print('Converting sentences to words id ... ')
        sentences_A = word_to_id(sentences_A, word2id)
        sentences_B = word_to_id(sentences_B, word2id)
        print('Converting sentences to words id completed')
        
        data = [sentences_A, sentences_B, relatedness, masks_A, masks_B]
        
        initializer = tf.random_normal_initializer(0.0, 0.2, dtype=tf.float32)
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = SiameseLSTM(config=config, sess=session, embeddings=embeddings, is_training=True)
        
        init = tf.global_variables_initializer()
        session.run(init)
        
        global_steps = 0
        for i in range(config['num_epoch']):
            global_steps = run_epoch(model, session, data, global_steps)
        
            
            



if __name__ == '__main__':
    train()