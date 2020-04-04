
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import json
import random
import pickle


from untitled4 import BLEU


class Seq2Seq_Model():
    def __init__(self, length_rnn, num_layers, dim_video_feat, embedding_size, 
                    learning_rate, word_to_idx, mode, max_gradient_norm, 
                    use_attention, beam_search, beam_size, 
                    max_encoder_steps, max_decoder_steps):
        tf.set_random_seed(6)
        np.random.seed(6)
        random.seed(6)

        self.length_rnn = length_rnn
        self.num_layers = num_layers
        self.dim_video_feat = dim_video_feat
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.word_to_idx = word_to_idx
        self.mode = mode
        self.max_gradient_norm = max_gradient_norm
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_encoder_steps = max_encoder_steps
        self.max_decoder_steps = max_decoder_steps

        self.vocab_size = len(self.word_to_idx)

        self.build_model()

    
    
    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_length):
       
        feed_dict = {self.encoder_inputs: encoder_inputs,
                      self.encoder_inputs_length: encoder_inputs_length,
                      self.decoder_targets: decoder_targets,
                      self.decoder_targets_length: decoder_targets_length,
                      self.keep_prob_placeholder: 0.8,
                      self.batch_size: len(encoder_inputs)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def eval(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_length):
  
        feed_dict = {self.encoder_inputs: encoder_inputs,
                      self.encoder_inputs_length: encoder_inputs_length,
                      self.decoder_targets: decoder_targets,
                      self.decoder_targets_length: decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def build_model(self):
        tf.set_random_seed(6)
        np.random.seed(6)
        random.seed(6)

        print ('Building model...')
        self.encoder_inputs = tf.placeholder(tf.float32, [None, None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

  
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
           
            encoder_inputs_flatten = tf.reshape(self.encoder_inputs, [-1, self.dim_video_feat])
            encoder_inputs_embedded = tf.layers.dense(encoder_inputs_flatten, self.embedding_size, use_bias=True)
            encoder_inputs_embedded = tf.reshape(encoder_inputs_embedded, [self.batch_size, self.max_encoder_steps, self.length_rnn])

           
            encoder_cell = self._create_rnn_cell()

        
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_inputs_embedded, 
                sequence_length=self.encoder_inputs_length, 
                dtype=tf.float32)

     
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            encoder_inputs_length = self.encoder_inputs_length

            if self.beam_search:
               
                print("Using beamsearch decoding...")
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_state = tf.contrib.framework.nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)

        
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

          
            projection_layer = tf.layers.Dense(units=self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=6))
            
       
            embedding_decoder = tf.Variable(tf.random_uniform([self.vocab_size, self.length_rnn], -0.1, 0.1, seed=6), name='embedding_decoder')


           
            decoder_cell = self._create_rnn_cell()

            if self.use_attention:
                
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.length_rnn, 
                    memory=encoder_outputs, 
                    normalize=True,
                    memory_sequence_length=encoder_inputs_length)

               
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell, 
                    attention_mechanism=attention_mechanism, 
                    attention_layer_size=self.length_rnn, 
                    name='Attention_Wrapper')

              
                decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            else:
                decoder_initial_state = encoder_state

            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, seed=6))

            
            ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
            decoder_inputs = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<bos>']), ending], 1)
          
            decoder_inputs_embedded = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

           
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=decoder_inputs_embedded, 
                sequence_length=self.decoder_targets_length, 
                time_major=False, name='training_helper')
           
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=training_helper, 
                initial_state=decoder_initial_state, 
                output_layer=output_layer)
           
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder, 
                impute_finished=True, 
                maximum_iterations=self.max_target_sequence_length)

            self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
            self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')

           
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.decoder_logits_train, 
                targets=self.decoder_targets, 
                weights=self.mask)

           
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<bos>']
            end_token = self.word_to_idx['<eos>']
            
          
            if self.beam_search:
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell, 
                    embedding=embedding_decoder,
                    start_tokens=start_tokens, 
                    end_token=end_token,
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_size,
                    output_layer=output_layer)
            else:

                inference_decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=embedding_decoder, 
                    start_tokens=start_tokens, 
                    end_token=end_token)

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell, 
                    helper=inference_decoding_helper, 
                    initial_state=decoder_initial_state, 
                    output_layer=output_layer)

            inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder, 
                maximum_iterations=self.max_decoder_steps)

            if self.beam_search:
                self.decoder_predict_decode = inference_decoder_outputs.predicted_ids
                self.decoder_predict_logits = inference_decoder_outputs.beam_search_decoder_output
            else:
                self.decoder_predict_decode = tf.expand_dims(inference_decoder_outputs.sample_id, -1)
                self.decoder_predict_logits = inference_decoder_outputs.rnn_output


        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

    def _create_rnn_cell(self):
        def new_rnn_cell():
            new_cell = tf.contrib.rnn.GRUCell(self.length_rnn)
            cell = tf.contrib.rnn.DropoutWrapper(new_cell, output_keep_prob=self.keep_prob_placeholder, seed=6)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell([new_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def infer(self, sess, encoder_inputs, encoder_inputs_length):
        
        feed_dict = {self.encoder_inputs: encoder_inputs,
                      self.encoder_inputs_length: encoder_inputs_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(encoder_inputs)}
        predict, logits = sess.run([self.decoder_predict_decode, self.decoder_predict_logits], feed_dict=feed_dict)
        return predict, logits

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
  
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

   
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

       
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

if __name__ == "__main__":

    np.random.seed(6)
    random.seed(6)
    tf.set_random_seed(6)

    test_video_feat_folder = sys.argv[1]
    testing_label_json_file = "./a.json"
    output_testset_filename = sys.argv[2]

    tf.app.flags.DEFINE_integer('length_rnn', 1024, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
    tf.app.flags.DEFINE_integer('dim_video_feat', 4096, 'Feature dimensions of each video frame')
    tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')

    tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
    tf.app.flags.DEFINE_integer('batch_size', 29, 'Batch size')
    tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Max. global gradient norm to clip')

    tf.app.flags.DEFINE_boolean('use_attention', True, 'Enable attention')
    
    tf.app.flags.DEFINE_boolean('beam_search', False, 'Enable beam search')
    tf.app.flags.DEFINE_integer('beam_size', 5, 'Size of beam search')

    tf.app.flags.DEFINE_integer('max_encoder_steps', 64, 'Max. steps of encoder')
    tf.app.flags.DEFINE_integer('max_decoder_steps', 15, 'Max. steps of decoder')

    tf.app.flags.DEFINE_integer('sample_size', 1450, 'Sampled data size of training epochs')
    tf.app.flags.DEFINE_integer('dim_video_frame', 80, 'Number of frame in each video')

    tf.app.flags.DEFINE_integer('num_epochs', 4, 'Maximum # of training epochs')
    tf.app.flags.DEFINE_string('model_dir', 'models/', 'Path to save model checkpoints')
    tf.app.flags.DEFINE_string('model_name', 's2s.ckpt', 'File name used for model checkpoints')

    FLAGS = tf.app.flags.FLAGS
    
    num_top_BLEU = 10
    top_BLEU = []

    print ('Reading pickle files...')
    word2index = pickle.load(open('word2index.obj', 'rb'))
    index2word = pickle.load(open('index2word.obj', 'rb'))
    video_IDs = pickle.load(open('video_IDs.obj', 'rb'))
    video_caption_dict = pickle.load(open('video_caption_dict.obj', 'rb'))
    video_feat_dict = pickle.load(open('video_feat_dict.obj', 'rb'))
    index2word_series = pd.Series(index2word)

    print ('Reading testing files...')
    test_video_feat_filenames = os.listdir(test_video_feat_folder)
    test_video_feat_filepaths = [(test_video_feat_folder + filename) for filename in test_video_feat_filenames]
    
    test_video_IDs = [filename[:-4] for filename in test_video_feat_filenames]

    test_video_feat_dict = {}
    for filepath in test_video_feat_filepaths:
        test_video_feat = np.load(filepath)
        
        sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
        test_video_feat = test_video_feat[sampled_video_frame]

        test_video_ID = filepath[: -4].replace(test_video_feat_folder, "")
        test_video_feat_dict[test_video_ID] = test_video_feat
    
            
     
    test_video_caption = json.load(open(testing_label_json_file, 'r'))

    with tf.Session() as sess:
        model = Seq2Seq_Model(
            length_rnn=FLAGS.length_rnn, 
            num_layers=FLAGS.num_layers, 
            dim_video_feat=FLAGS.dim_video_feat, 
            embedding_size=FLAGS.embedding_size, 
            learning_rate=FLAGS.learning_rate, 
            word_to_idx=word2index, 
            mode='train', 
            max_gradient_norm=FLAGS.max_gradient_norm, 
            use_attention=FLAGS.use_attention, 
            beam_search=FLAGS.beam_search, 
            beam_size=FLAGS.beam_size,
            max_encoder_steps=FLAGS.max_encoder_steps, 
            max_decoder_steps=FLAGS.max_decoder_steps
        )
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Created new model parameters..')
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        for epoch in range(FLAGS.num_epochs):
            start_time = time.time()

            sampled_ID_caption = []
            for ID in video_IDs:
                sampled_caption = random.sample(video_caption_dict[ID], 1)[0]
                sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
                sampled_video_feat = video_feat_dict[ID][sampled_video_frame]
                sampled_ID_caption.append((sampled_video_feat, sampled_caption))

       
            random.shuffle(sampled_ID_caption)

            for batch_start, batch_end in zip(range(0, FLAGS.sample_size, FLAGS.batch_size), range(FLAGS.batch_size, FLAGS.sample_size, FLAGS.batch_size)):
                print ("%04d/%04d" %(batch_end, FLAGS.sample_size), end='\r')

                batch_sampled_ID_caption = sampled_ID_caption[batch_start : batch_end]
                batch_video_feats = [elements[0] for elements in batch_sampled_ID_caption]
                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size
             
                batch_captions = np.array(["<bos> "+ elements[1] for elements in batch_sampled_ID_caption])

                for index, caption in enumerate(batch_captions):
                    caption_words = caption.lower().split(" ")
                    if len(caption_words) < FLAGS.max_decoder_steps:
                        batch_captions[index] = batch_captions[index] + " <eos>"
                    else:
                        new_caption = ""
                        for i in range(FLAGS.max_decoder_steps - 1):
                            new_caption = new_caption + caption_words[i] + " "
                        batch_captions[index] = new_caption + "<eos>"

                batch_captions_words_index = []
                for caption in batch_captions:
                    words_index = []
                    for caption_words in caption.lower().split(' '):
                        if caption_words in word2index:
                            words_index.append(word2index[caption_words])
                        else:
                            words_index.append(word2index['<unk>'])
                    batch_captions_words_index.append(words_index)

                batch_captions_matrix = pad_sequences(batch_captions_words_index, padding='post', maxlen=FLAGS.max_decoder_steps)
                
                batch_captions_length = [len(x) for x in batch_captions_matrix]
               
                loss, summary = model.train(
                    sess, 
                    batch_video_feats, 
                    batch_video_frame, 
                    batch_captions_matrix, 
                    batch_captions_length)
            
            print()
               
            test_captions = []
            for batch_start, batch_end in zip(range(0, len(test_video_IDs) + FLAGS.batch_size, FLAGS.batch_size), range(FLAGS.batch_size, len(test_video_IDs) + FLAGS.batch_size, FLAGS.batch_size)):
                print ("%04d/%04d" %(batch_end, FLAGS.sample_size), end='\r')
                if batch_end < len(test_video_IDs):
                    batch_sampled_ID = np.array(test_video_IDs[batch_start : batch_end])
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sampled_ID]
                else:
                    batch_sampled_ID = test_video_IDs[batch_start : batch_end]
                    for _ in range(batch_end - len(test_video_IDs)):
                        batch_sampled_ID.append(test_video_IDs[-1])
                    batch_sampled_ID = np.array(batch_sampled_ID)
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sampled_ID]

                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size 

                batch_caption_words_index, logits = model.infer(
                    sess, 
                    batch_video_feats, 
                    batch_video_frame) 

                if batch_end < len(test_video_IDs):
                    batch_caption_words_index = batch_caption_words_index
                else:
                    batch_caption_words_index = batch_caption_words_index[:len(test_video_IDs) - batch_start]

                for index, test_caption_words_index in enumerate(batch_caption_words_index):

                    if FLAGS.beam_search:
                        logits = np.array(logits).reshape(-1, FLAGS.beam_size)
                        max_logits_index = np.argmax(np.sum(logits, axis=0))
                        predict_list = np.ndarray.tolist(test_caption_words_index[0, :, max_logits_index])
                        predict_seq = [index2word[idx] for idx in predict_list]
                        test_caption_words = predict_seq
                    else:
                        test_caption_words_index = np.array(test_caption_words_index).reshape(-1)
                        test_caption_words = index2word_series[test_caption_words_index]
                        test_caption = ' '.join(test_caption_words) 

                    test_caption = ' '.join(test_caption_words)
                    test_caption = test_caption.replace('<bos> ', '')
                    test_caption = test_caption.replace('<eos>', '')
                    test_caption = test_caption.replace(' <eos>', '')
                    test_caption = test_caption.replace('<pad> ', '')
                    test_caption = test_caption.replace(' <pad>', '')
                    test_caption = test_caption.replace(' <unk>', '')
                    test_caption = test_caption.replace('<unk> ', '')

                    if (test_caption == ""):
                        test_caption = '.'

                    if batch_sampled_ID[index] in ["klteYv1Uv9A_27_33.avi", "UbmZAe5u5FI_132_141.avi", "wkgGxsuNVSg_34_41.avi", "JntMAcTlOF0_50_70.avi", "tJHUH9tpqPg_113_118.avi"]:
                        print(batch_sampled_ID[index], test_caption)
                    test_captions.append(test_caption)
                        
            df = pd.DataFrame(np.array([test_video_IDs, test_captions]).T)
            df.to_csv(output_testset_filename, index=False, header=False)

            result = {}
            with open(output_testset_filename, 'r') as f:
                for line in f:
                    line = line.rstrip()
                    test_id, caption = line.split(',')
                    result[test_id] = caption
                    
            bleu=[]
            for item in test_video_caption:
                score_per_video = []
                captions = [x.rstrip('.') for x in item['caption']]
                score_per_video.append(BLEU(result[item['id']],captions,True))
                bleu.append(score_per_video[0])
            average = sum(bleu) / len(bleu)

            if (len(top_BLEU) < num_top_BLEU):
                top_BLEU.append(average)
                print ("Saving model with BLEU@1: %.4f ..." %(average))
                model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            else:
                if (average > min(top_BLEU)):
                    top_BLEU.remove(min(top_BLEU))
                    top_BLEU.append(average)
                    print ("Saving model with BLEU@1: %.4f ..." %(average))
                    model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            top_BLEU.sort(reverse=True)
            print ("Top [%d] BLEU: " %(num_top_BLEU), ["%.4f" % x for x in top_BLEU])

        
            print ("Epoch %d/%d, loss: %.6f, Avg. BLEU@1: %.6f, Elapsed time: %.2fs" %(epoch, FLAGS.num_epochs, loss, average, (time.time() - start_time)))
