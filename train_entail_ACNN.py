import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle
from sklearn.preprocessing import normalize
from scipy.stats import mode

from load_data import load_SNLI_dataset_with_extra, load_word2vec, load_word2vec_to_init, extend_word2vec_lowercase
from common_functions import Conv_for_Pair,dropout_layer, store_model_to_file, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, ABCNN, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
'''
1, use SVM outside
drop0.05, reach 0.86345177665
'''

def evaluate_lenet5(learning_rate=0.02, n_epochs=4, L2_weight=0.0000001, extra_size=4, use_svm=False, drop_p=0.1, div_weight=0.00001, emb_size=300, batch_size=50, filter_size=[3,3], maxSentLen=40, hidden_size=[300,300], margin =0.1, comment='five copies, sum&majority'):

    model_options = locals().copy()
    print "model options", model_options

    first_seed=1234
    np.random.seed(first_seed)
    first_rng = np.random.RandomState(first_seed)    #random seed, control the model generates the same results
    first_srng = RandomStreams(first_rng.randint(999999))

    second_seed=2345
    np.random.seed(second_seed)
    second_rng = np.random.RandomState(second_seed)    #random seed, control the model generates the same results
    second_srng = RandomStreams(second_rng.randint(888888))

    third_seed=3456
    np.random.seed(third_seed)
    third_rng = np.random.RandomState(third_seed)    #random seed, control the model generates the same results
    third_srng = RandomStreams(third_rng.randint(777777))

    fourth_seed=4567
    np.random.seed(fourth_seed)
    fourth_rng = np.random.RandomState(fourth_seed)    #random seed, control the model generates the same results
    fourth_srng = RandomStreams(fourth_rng.randint(666666))

    fifth_seed=5678
    np.random.seed(fifth_seed)
    fifth_rng = np.random.RandomState(fifth_seed)    #random seed, control the model generates the same results
    fifth_srng = RandomStreams(fifth_rng.randint(555555))
        
    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_extra, all_labels, word2id  =load_SNLI_dataset_with_extra(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    train_sents_l=np.asarray(all_sentences_l[0], dtype='int32')
    dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
#     train_sents_l = np.concatenate((train_sents_l, dev_sents_l), axis=0)
    test_sents_l=np.asarray(all_sentences_l[2], dtype='int32')

    train_masks_l=np.asarray(all_masks_l[0], dtype=theano.config.floatX)
    dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
#     train_masks_l = np.concatenate((train_masks_l, dev_masks_l), axis=0)
    test_masks_l=np.asarray(all_masks_l[2], dtype=theano.config.floatX)

    train_sents_r=np.asarray(all_sentences_r[0], dtype='int32')
    dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
#     train_sents_r = np.concatenate((train_sents_r, dev_sents_r), axis=0)
    test_sents_r=np.asarray(all_sentences_r[2] , dtype='int32')

    train_masks_r=np.asarray(all_masks_r[0], dtype=theano.config.floatX)
    dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
#     train_masks_r = np.concatenate((train_masks_r, dev_masks_r), axis=0)
    test_masks_r=np.asarray(all_masks_r[2], dtype=theano.config.floatX)

    train_extra=np.asarray(all_extra[0], dtype=theano.config.floatX)
    dev_extra=np.asarray(all_extra[1], dtype=theano.config.floatX)
    test_extra=np.asarray(all_extra[2], dtype=theano.config.floatX)

    train_labels_store=np.asarray(all_labels[0], dtype='int32')
    dev_labels_store=np.asarray(all_labels[1], dtype='int32')
#     train_labels_store = np.concatenate((train_labels_store, dev_labels_store), axis=0)
    test_labels_store=np.asarray(all_labels[2], dtype='int32')

    train_size=len(train_labels_store)
    dev_size=len(dev_labels_store)
    test_size=len(test_labels_store)
    print 'train size: ', train_size, ' dev size: ', dev_size, ' test size: ', test_size

    vocab_size=len(word2id)+1


    rand_values=first_rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
#     word2vec =extend_word2vec_lowercase(word2vec)
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
#     normed_matrix = normalize(rand_values, axis=0, norm='l2')
    first_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable

    second_rand_values=second_rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    second_rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    second_rand_values=load_word2vec_to_init(second_rand_values, id2word, word2vec)
    second_embeddings=theano.shared(value=np.array(second_rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable

    third_rand_values=third_rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    third_rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    third_rand_values=load_word2vec_to_init(third_rand_values, id2word, word2vec)
    third_embeddings=theano.shared(value=np.array(third_rand_values,dtype=theano.config.floatX), borrow=True)

    fourth_rand_values=fourth_rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    fourth_rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    fourth_rand_values=load_word2vec_to_init(fourth_rand_values, id2word, word2vec)
    fourth_embeddings=theano.shared(value=np.array(fourth_rand_values,dtype=theano.config.floatX), borrow=True)

    fifth_rand_values=fifth_rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    fifth_rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    fifth_rand_values=load_word2vec_to_init(fifth_rand_values, id2word, word2vec)
    fifth_embeddings=theano.shared(value=np.array(fifth_rand_values,dtype=theano.config.floatX), borrow=True)

    #now, start to build the input form of the model
    train_flag = T.iscalar()
    first_sents_ids_l=T.imatrix()
    first_sents_mask_l=T.fmatrix()
    first_sents_ids_r=T.imatrix()
    first_sents_mask_r=T.fmatrix()
    first_labels=T.ivector()
    second_sents_ids_l=T.imatrix()
    second_sents_mask_l=T.fmatrix()
    second_sents_ids_r=T.imatrix()
    second_sents_mask_r=T.fmatrix()
    second_labels=T.ivector()
    third_sents_ids_l=T.imatrix()
    third_sents_mask_l=T.fmatrix()
    third_sents_ids_r=T.imatrix()
    third_sents_mask_r=T.fmatrix()
    third_labels=T.ivector()
    fourth_sents_ids_l=T.imatrix()
    fourth_sents_mask_l=T.fmatrix()
    fourth_sents_ids_r=T.imatrix()
    fourth_sents_mask_r=T.fmatrix()
    fourth_labels=T.ivector()
    fifth_sents_ids_l=T.imatrix()
    fifth_sents_mask_l=T.fmatrix()
    fifth_sents_ids_r=T.imatrix()
    fifth_sents_mask_r=T.fmatrix()
    fifth_labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    def common_input(emb_matrix, sent_ids):
        return emb_matrix[sent_ids.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    first_common_input_l=common_input(first_embeddings, first_sents_ids_l)#embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    first_common_input_r=common_input(first_embeddings, first_sents_ids_r)#embeddings[sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    second_common_input_l=common_input(second_embeddings, second_sents_ids_l)#second_embeddings[second_sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    second_common_input_r=common_input(second_embeddings, second_sents_ids_r)#second_embeddings[second_sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    third_common_input_l=common_input(third_embeddings, third_sents_ids_l)#third_embeddings[third_sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    third_common_input_r=common_input(third_embeddings, third_sents_ids_r)#third_embeddings[third_sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    fourth_common_input_l=common_input(fourth_embeddings, fourth_sents_ids_l)#fourth_embeddings[fourth_sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    fourth_common_input_r=common_input(fourth_embeddings, fourth_sents_ids_r)#fourth_embeddings[fourth_sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    fifth_common_input_l=common_input(fifth_embeddings, fifth_sents_ids_l)#fifth_embeddings[fifth_sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    fifth_common_input_r=common_input(fifth_embeddings, fifth_sents_ids_r)#fifth_embeddings[fifth_sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)




    gate_filter_shape=(hidden_size[0], 1, emb_size, 1)
    def create_CNN_params(rng):
        conv_W_2_pre, conv_b_2_pre=create_conv_para(rng, filter_shape=gate_filter_shape)
        conv_W_2_gate, conv_b_2_gate=create_conv_para(rng, filter_shape=gate_filter_shape)
        conv_W_2, conv_b_2=create_conv_para(rng, filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[0]))
        conv_W_2_context, conv_b_2_context=create_conv_para(rng, filter_shape=(hidden_size[1], 1, hidden_size[0], 1))
        return conv_W_2_pre, conv_b_2_pre,conv_W_2_gate, conv_b_2_gate,conv_W_2, conv_b_2,conv_W_2_context, conv_b_2_context
    
    first_conv_W_pre, first_conv_b_pre,first_conv_W_gate, first_conv_b_gate,first_conv_W, first_conv_b,first_conv_W_context, first_conv_b_context = create_CNN_params(first_rng)
    second_conv_W_pre, second_conv_b_pre,second_conv_W_gate, second_conv_b_gate,second_conv_W, second_conv_b,second_conv_W_context, second_conv_b_context = create_CNN_params(second_rng)
    third_conv_W_pre, third_conv_b_pre,third_conv_W_gate, third_conv_b_gate,third_conv_W, third_conv_b,third_conv_W_context, third_conv_b_context = create_CNN_params(third_rng)
    fourth_conv_W_pre, fourth_conv_b_pre,fourth_conv_W_gate, fourth_conv_b_gate,fourth_conv_W, fourth_conv_b,fourth_conv_W_context, fourth_conv_b_context = create_CNN_params(fourth_rng)
    fifth_conv_W_pre, fifth_conv_b_pre,fifth_conv_W_gate, fifth_conv_b_gate,fifth_conv_W, fifth_conv_b,fifth_conv_W_context, fifth_conv_b_context = create_CNN_params(fifth_rng)

    '''
    dropout paras
    '''
    def dropout_group(rng, conv_W_2_pre, conv_W_2_gate, conv_W_2, conv_W_2_context):
        drop_conv_W_2_pre = dropout_layer(rng, conv_W_2_pre, drop_p, train_flag)
        drop_conv_W_2_gate = dropout_layer(rng, conv_W_2_gate, drop_p, train_flag)
        drop_conv_W_2 = dropout_layer(rng, conv_W_2, drop_p, train_flag)
        drop_conv_W_2_context = dropout_layer(rng, conv_W_2_context, drop_p, train_flag)
        return drop_conv_W_2_pre,drop_conv_W_2_gate,drop_conv_W_2,drop_conv_W_2_context
    drop_first_conv_W_pre,drop_first_conv_W_gate,drop_first_conv_W,drop_first_conv_W_context = dropout_group(first_srng, first_conv_W_pre, first_conv_W_gate, first_conv_W, first_conv_W_context)
    drop_second_conv_W_pre,drop_second_conv_W_gate,drop_second_conv_W,drop_second_conv_W_context = dropout_group(second_srng, second_conv_W_pre, second_conv_W_gate, second_conv_W, second_conv_W_context)
    drop_third_conv_W_pre,drop_third_conv_W_gate,drop_third_conv_W,drop_third_conv_W_context = dropout_group(third_srng, third_conv_W_pre, third_conv_W_gate, third_conv_W, third_conv_W_context)
    drop_fourth_conv_W_pre,drop_fourth_conv_W_gate,drop_fourth_conv_W,drop_fourth_conv_W_context = dropout_group(fourth_srng, fourth_conv_W_pre, fourth_conv_W_gate, fourth_conv_W, fourth_conv_W_context)
    drop_fifth_conv_W_pre,drop_fifth_conv_W_gate,drop_fifth_conv_W,drop_fifth_conv_W_context = dropout_group(fifth_srng, fifth_conv_W_pre, fifth_conv_W_gate, fifth_conv_W, fifth_conv_W_context)
    
    first_NN_para=[#conv_W, conv_b,
            first_conv_W_pre, first_conv_b_pre,
            first_conv_W_gate, first_conv_b_gate,
            first_conv_W, first_conv_b,first_conv_W_context]
    second_NN_para=[
            second_conv_W_pre, second_conv_b_pre,
            second_conv_W_gate, second_conv_b_gate,
            second_conv_W, second_conv_b,second_conv_W_context]

    third_NN_para=[
            third_conv_W_pre, third_conv_b_pre,
            third_conv_W_gate, third_conv_b_gate,
            third_conv_W, third_conv_b,third_conv_W_context]
    fourth_NN_para=[
            fourth_conv_W_pre, fourth_conv_b_pre,
            fourth_conv_W_gate, fourth_conv_b_gate,
            fourth_conv_W, fourth_conv_b,fourth_conv_W_context]

    fifth_NN_para=[
            fifth_conv_W_pre, fifth_conv_b_pre,
            fifth_conv_W_gate, fifth_conv_b_gate,
            fifth_conv_W, fifth_conv_b,fifth_conv_W_context]
    '''
    first classifier
    '''
    def classifier(rng,common_input_l,common_input_r,sents_mask_l, sents_mask_r,drop_conv_W_2_pre,conv_b_2_pre,drop_conv_W_2_gate,conv_b_2_gate,drop_conv_W_2,conv_b_2,drop_conv_W_2_context,
                   conv_b_2_context,labels):
        conv_layer_2_gate_l = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_l,
                 mask_matrix = sents_mask_l,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=gate_filter_shape,
                 W=drop_conv_W_2_pre, b=conv_b_2_pre,
                 W_gate =drop_conv_W_2_gate, b_gate=conv_b_2_gate )
        conv_layer_2_gate_r = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input_r,
                 mask_matrix = sents_mask_r,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=gate_filter_shape,
                 W=drop_conv_W_2_pre, b=conv_b_2_pre,
                 W_gate =drop_conv_W_2_gate, b_gate=conv_b_2_gate )
    
        l_input_4_att = conv_layer_2_gate_l.output_tensor3#conv_layer_2_gate_l.masked_conv_out_sigmoid*conv_layer_2_pre_l.masked_conv_out+(1.0-conv_layer_2_gate_l.masked_conv_out_sigmoid)*common_input_l
        r_input_4_att = conv_layer_2_gate_r.output_tensor3#conv_layer_2_gate_r.masked_conv_out_sigmoid*conv_layer_2_pre_r.masked_conv_out+(1.0-conv_layer_2_gate_r.masked_conv_out_sigmoid)*common_input_r
    
        conv_layer_2 = Conv_for_Pair(rng,
                origin_input_tensor3=common_input_l,
                origin_input_tensor3_r = common_input_r,
                input_tensor3=l_input_4_att,
                input_tensor3_r = r_input_4_att,
                 mask_matrix = sents_mask_l,
                 mask_matrix_r = sents_mask_r,
                 image_shape=(batch_size, 1, hidden_size[0], maxSentLen),
                 image_shape_r = (batch_size, 1, hidden_size[0], maxSentLen),
                 filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[0]),
                 filter_shape_context=(hidden_size[1], 1,hidden_size[0], 1),
                 W=drop_conv_W_2, b=conv_b_2,
                 W_context=drop_conv_W_2_context, b_context=conv_b_2_context)
        attentive_sent_embeddings_l_2 = conv_layer_2.attentive_maxpool_vec_l
        attentive_sent_embeddings_r_2 = conv_layer_2.attentive_maxpool_vec_r
        # attentive_sent_sumpool_l_2 = conv_layer_2.attentive_sumpool_vec_l
        # attentive_sent_sumpool_r_2 = conv_layer_2.attentive_sumpool_vec_r
    
        HL_layer_1_input = T.concatenate([attentive_sent_embeddings_l_2,attentive_sent_embeddings_r_2, attentive_sent_embeddings_l_2*attentive_sent_embeddings_r_2],axis=1)
    
        HL_layer_1_input_size = hidden_size[1]*3#+extra_size#+(maxSentLen*2+10*2)#+hidden_size[1]*3+1
    
        HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size[0], activation=T.nnet.relu)
        HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=hidden_size[0], n_out=hidden_size[0], activation=T.nnet.relu)

        LR_input_size=HL_layer_1_input_size+2*hidden_size[0]
        U_a = create_ensemble_para(rng, 3, LR_input_size) # the weight matrix hidden_size*2
        LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
        LR_para=[U_a, LR_b]
    
        LR_input=T.tanh(T.concatenate([HL_layer_1_input, HL_layer_1.output, HL_layer_2.output],axis=1))
        layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=3, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
        loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.

        return loss, LR_para+HL_layer_1.params+HL_layer_2.params, layer_LR.p_y_given_x, layer_LR.errors(labels)
    
    first_loss, first_classifier_params, first_test_distr, first_error = classifier(first_rng,first_common_input_l,first_common_input_r,first_sents_mask_l,first_sents_mask_r,drop_first_conv_W_pre,first_conv_b_pre,
                                                                       drop_first_conv_W_gate,first_conv_b_gate,drop_first_conv_W,first_conv_b,drop_first_conv_W_context,
                                                                       first_conv_b_context, first_labels)
    second_loss, second_classifier_params, second_test_distr, second_error = classifier(second_rng,second_common_input_l,second_common_input_r,second_sents_mask_l,second_sents_mask_r,drop_second_conv_W_pre,second_conv_b_pre,
                                                                       drop_second_conv_W_gate,second_conv_b_gate,drop_second_conv_W,second_conv_b,drop_second_conv_W_context,
                                                                       second_conv_b_context, second_labels)
    third_loss, third_classifier_params, third_test_distr, third_error = classifier(third_rng,third_common_input_l,third_common_input_r,third_sents_mask_l,third_sents_mask_r,drop_third_conv_W_pre,third_conv_b_pre,
                                                                       drop_third_conv_W_gate,third_conv_b_gate,drop_third_conv_W,third_conv_b,drop_third_conv_W_context,
                                                                       third_conv_b_context, third_labels)
    fourth_loss, fourth_classifier_params, fourth_test_distr, fourth_error = classifier(fourth_rng,fourth_common_input_l,fourth_common_input_r,fourth_sents_mask_l,fourth_sents_mask_r,drop_fourth_conv_W_pre,fourth_conv_b_pre,
                                                                       drop_fourth_conv_W_gate,fourth_conv_b_gate,drop_fourth_conv_W,fourth_conv_b,drop_fourth_conv_W_context,
                                                                       fourth_conv_b_context, fourth_labels)
    fifth_loss, fifth_classifier_params, fifth_test_distr, fifth_error = classifier(fifth_rng,fifth_common_input_l,fifth_common_input_r,fifth_sents_mask_l,fifth_sents_mask_r,drop_fifth_conv_W_pre,fifth_conv_b_pre,
                                                                       drop_fifth_conv_W_gate,fifth_conv_b_gate,drop_fifth_conv_W,fifth_conv_b,drop_fifth_conv_W_context,
                                                                       fifth_conv_b_context, fifth_labels)


    '''
    testing, labels == second_labels
    '''
    all_prop_distr = first_test_distr+second_test_distr+third_test_distr+fourth_test_distr+fifth_test_distr
    first_preds = T.argmax(first_test_distr, axis=1).dimshuffle('x',0) #(1, batch)
    second_preds = T.argmax(second_test_distr, axis=1).dimshuffle('x',0) #(1, batch)
    third_preds = T.argmax(third_test_distr, axis=1).dimshuffle('x',0) #(1, batch)
    fourth_preds = T.argmax(fourth_test_distr, axis=1).dimshuffle('x',0) #(1, batch)
    fifth_preds = T.argmax(fifth_test_distr, axis=1).dimshuffle('x',0) #(1, batch)
    overall_preds = T.concatenate([first_preds,second_preds,third_preds,fourth_preds,fifth_preds], axis=0) #(5, batch)
    all_error = T.mean(T.neq(T.argmax(all_prop_distr, axis=1), first_labels))



#     neg_labels = T.where( labels < 2, 2, labels-1)
#     loss2=-T.mean(T.log(1.0/(1.0+layer_LR.p_y_given_x))[T.arange(neg_labels.shape[0]), neg_labels])

    # rank loss
    # entail_prob_batch = T.nnet.softmax(layer_LR.before_softmax.T)[2] #batch
    # entail_ids = elementwise_is_two(labels)
    # entail_probs = entail_prob_batch[entail_ids.nonzero()]
    # non_entail_probs = entail_prob_batch[(1-entail_ids).nonzero()]
    #
    # repeat_entail = T.extra_ops.repeat(entail_probs, non_entail_probs.shape[0], axis=0)
    # repeat_non_entail = T.extra_ops.repeat(non_entail_probs.dimshuffle('x',0), entail_probs.shape[0], axis=0).flatten()
    # loss2 = -T.mean(T.log(entail_probs))#T.mean(T.maximum(0.0, margin-repeat_entail+repeat_non_entail))

    # zero_matrix = T.zeros((batch_size, 3))
    # filled_zero_matrix = T.set_subtensor(zero_matrix[T.arange(batch_size), labels], 1.0)
    # prob_batch_posi = layer_LR.p_y_given_x[filled_zero_matrix.nonzero()]
    # prob_batch_nega = layer_LR.p_y_given_x[(1-filled_zero_matrix).nonzero()]
    #
    # repeat_posi = T.extra_ops.repeat(prob_batch_posi, prob_batch_nega.shape[0], axis=0)
    # repeat_nega = T.extra_ops.repeat(prob_batch_nega.dimshuffle('x',0), prob_batch_posi.shape[0], axis=0).flatten()
    # loss2 = T.mean(T.maximum(0.0, margin-repeat_posi+repeat_nega))

    
    first_params = [first_embeddings]+first_NN_para+first_classifier_params
    second_params = [second_embeddings]+second_NN_para+second_classifier_params
    third_params = [third_embeddings]+third_NN_para+third_classifier_params
    fourth_params = [fourth_embeddings]+fourth_NN_para+fourth_classifier_params
    fifth_params = [fifth_embeddings]+fifth_NN_para+fifth_classifier_params
    
    params = first_params+second_params+third_params+fourth_params+fifth_params
    
#     L2_reg =L2norm_paraList([embeddings,HL_layer_1.W, HL_layer_2.W])

#     diversify_reg= (Diversify_Reg(conv_W_2_pre_to_matrix)+Diversify_Reg(conv_W_2_gate_to_matrix)+
#                     Diversify_Reg(conv_W_2_to_matrix)+Diversify_Reg(conv_W_2_context_to_matrix))

    cost=first_loss+second_loss+third_loss+fourth_loss+fifth_loss#+0.1*loss2#+loss2#+L2_weight*L2_reg

    
    first_updates =   Gradient_Cost_Para(first_loss,first_params, learning_rate)
    second_updates =   Gradient_Cost_Para(second_loss,second_params, learning_rate)
    third_updates =   Gradient_Cost_Para(third_loss,third_params, learning_rate)
    fourth_updates =   Gradient_Cost_Para(fourth_loss,fourth_params, learning_rate)
    fifth_updates =   Gradient_Cost_Para(fifth_loss,fifth_params, learning_rate)
    updates = first_updates+second_updates+third_updates+fourth_updates+fifth_updates
    
    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([train_flag,first_sents_ids_l, first_sents_mask_l, first_sents_ids_r, first_sents_mask_r, first_labels,
                                   second_sents_ids_l,second_sents_mask_l,second_sents_ids_r,second_sents_mask_r,second_labels,
                                   third_sents_ids_l,third_sents_mask_l,third_sents_ids_r,third_sents_mask_r,third_labels,
                                   fourth_sents_ids_l,fourth_sents_mask_l,fourth_sents_ids_r,fourth_sents_mask_r,fourth_labels,
                                   fifth_sents_ids_l,fifth_sents_mask_l,fifth_sents_ids_r,fifth_sents_mask_r,fifth_labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
#     train_model_pred = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,extra,labels,
#                                         second_sents_ids_l,second_sents_mask_l,second_sents_ids_r,second_sents_mask_r,second_labels], [LR_input, labels], allow_input_downcast=True, on_unused_input='ignore')
# 
#     dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,extra, labels,
#                                  second_sents_ids_l,second_sents_mask_l,second_sents_ids_r,second_sents_mask_r,second_labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([train_flag,first_sents_ids_l, first_sents_mask_l, first_sents_ids_r, first_sents_mask_r, first_labels,
                                  second_sents_ids_l,second_sents_mask_l,second_sents_ids_r,second_sents_mask_r,second_labels,
                                  third_sents_ids_l,third_sents_mask_l,third_sents_ids_r,third_sents_mask_r,third_labels,
                                  fourth_sents_ids_l,fourth_sents_mask_l,fourth_sents_ids_r,fourth_sents_mask_r,fourth_labels,
                                  fifth_sents_ids_l,fifth_sents_mask_l,fifth_sents_ids_r,fifth_sents_mask_r,fifth_labels], [first_error, second_error, third_error, fourth_error, fifth_error, all_error,overall_preds], allow_input_downcast=True, on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False

    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
    n_dev_batches=dev_size/batch_size
    dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]


    max_acc_dev=0.0
    max_acc_test=0.0
    max_acc_svm=0.0
    max_acc_lr=0.0
    max_majority_acc=0.0

    cost_i=0.0
    first_train_indices = range(train_size)
    second_train_indices = range(train_size)
    third_train_indices = range(train_size)
    fourth_train_indices = range(train_size)
    fifth_train_indices = range(train_size)
    while epoch < n_epochs:
        epoch = epoch + 1

        random.Random(100).shuffle(first_train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        random.Random(200).shuffle(second_train_indices)
        random.Random(300).shuffle(third_train_indices)
        random.Random(400).shuffle(fourth_train_indices)
        random.Random(500).shuffle(fifth_train_indices)
        iter_accu=0

        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            first_train_id_batch = first_train_indices[batch_id:batch_id+batch_size]
            second_train_id_batch = second_train_indices[batch_id:batch_id+batch_size]
            third_train_id_batch = third_train_indices[batch_id:batch_id+batch_size]
            fourth_train_id_batch = fourth_train_indices[batch_id:batch_id+batch_size]
            fifth_train_id_batch = fifth_train_indices[batch_id:batch_id+batch_size]
            cost_i+= train_model(
                                1,
                                train_sents_l[first_train_id_batch],
                                train_masks_l[first_train_id_batch],
                                train_sents_r[first_train_id_batch],
                                train_masks_r[first_train_id_batch],
                                train_labels_store[first_train_id_batch],
                                
                                train_sents_l[second_train_id_batch],
                                train_masks_l[second_train_id_batch],
                                train_sents_r[second_train_id_batch],
                                train_masks_r[second_train_id_batch],
                                train_labels_store[second_train_id_batch],
                                
                                train_sents_l[third_train_id_batch],
                                train_masks_l[third_train_id_batch],
                                train_sents_r[third_train_id_batch],
                                train_masks_r[third_train_id_batch],
                                train_labels_store[third_train_id_batch],
                                
                                train_sents_l[fourth_train_id_batch],
                                train_masks_l[fourth_train_id_batch],
                                train_sents_r[fourth_train_id_batch],
                                train_masks_r[fourth_train_id_batch],
                                train_labels_store[fourth_train_id_batch],
                                
                                train_sents_l[fifth_train_id_batch],
                                train_masks_l[fifth_train_id_batch],
                                train_sents_r[fifth_train_id_batch],
                                train_masks_r[fifth_train_id_batch],
                                train_labels_store[fifth_train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%int(2000*(50.0 / batch_size))==0:
#             if iter%int(200*(50.0 / batch_size))==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()
            # if epoch >=3 and iter >= len(train_batch_start)*2.0/3 and iter%500==0:
            #     print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
            #     past_time = time.time()

#                 error_sum=0.0
#                 for dev_batch_id in dev_batch_start: # for each test batch
#                     error_i=dev_model(
#                                 dev_sents_l[dev_batch_id:dev_batch_id+batch_size],
#                                 dev_masks_l[dev_batch_id:dev_batch_id+batch_size],
#                                 dev_sents_r[dev_batch_id:dev_batch_id+batch_size],
#                                 dev_masks_r[dev_batch_id:dev_batch_id+batch_size],
#                                 dev_labels_store[dev_batch_id:dev_batch_id+batch_size]
#                                 )
#
#                     error_sum+=error_i
#                 dev_accuracy=1.0-error_sum/(len(dev_batch_start))
#                 if dev_accuracy > max_acc_dev:
#                     max_acc_dev=dev_accuracy
#                     print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev
                    #best dev model, do test
                error_sum_1=0.0
                error_sum_2=0.0
                error_sum_3=0.0
                error_sum_4=0.0
                error_sum_5=0.0
                error_both=0.0
                ys=[]
                gold_ys=[]
                for test_batch_id in test_batch_start: # for each test batch
                    error_1, error_2, error_3, error_4, error_5, both_error_batch, batch_ys=test_model(
                            0,
                            test_sents_l[test_batch_id:test_batch_id+batch_size],
                            test_masks_l[test_batch_id:test_batch_id+batch_size],
                            test_sents_r[test_batch_id:test_batch_id+batch_size],
                            test_masks_r[test_batch_id:test_batch_id+batch_size],
                            test_labels_store[test_batch_id:test_batch_id+batch_size],
                            
                            test_sents_l[test_batch_id:test_batch_id+batch_size],
                            test_masks_l[test_batch_id:test_batch_id+batch_size],
                            test_sents_r[test_batch_id:test_batch_id+batch_size],
                            test_masks_r[test_batch_id:test_batch_id+batch_size],
                            test_labels_store[test_batch_id:test_batch_id+batch_size],
                            
                            test_sents_l[test_batch_id:test_batch_id+batch_size],
                            test_masks_l[test_batch_id:test_batch_id+batch_size],
                            test_sents_r[test_batch_id:test_batch_id+batch_size],
                            test_masks_r[test_batch_id:test_batch_id+batch_size],
                            test_labels_store[test_batch_id:test_batch_id+batch_size],

                            test_sents_l[test_batch_id:test_batch_id+batch_size],
                            test_masks_l[test_batch_id:test_batch_id+batch_size],
                            test_sents_r[test_batch_id:test_batch_id+batch_size],
                            test_masks_r[test_batch_id:test_batch_id+batch_size],
                            test_labels_store[test_batch_id:test_batch_id+batch_size],
                            
                            test_sents_l[test_batch_id:test_batch_id+batch_size],
                            test_masks_l[test_batch_id:test_batch_id+batch_size],
                            test_sents_r[test_batch_id:test_batch_id+batch_size],
                            test_masks_r[test_batch_id:test_batch_id+batch_size],
                            test_labels_store[test_batch_id:test_batch_id+batch_size]
                            )

                    error_sum_1+=error_1
                    error_sum_2+=error_2
                    error_sum_3+=error_3
                    error_sum_4+=error_4
                    error_sum_5+=error_5
                    error_both+=both_error_batch
                    ys.append(batch_ys)
                    gold_ys.append(test_labels_store[test_batch_id:test_batch_id+batch_size])
                test_acc_1=1.0-error_sum_1/(len(test_batch_start))
                test_acc_2=1.0-error_sum_2/(len(test_batch_start))
                test_acc_3=1.0-error_sum_3/(len(test_batch_start))
                test_acc_4=1.0-error_sum_4/(len(test_batch_start))
                test_acc_5=1.0-error_sum_5/(len(test_batch_start))
                
                all_ys = np.concatenate(ys, axis=1) #(5, test_size)
                gold_ys = np.concatenate(gold_ys)
                majority_ys= mode(np.transpose(all_ys), axis=-1)[0][:,0]
                majority_acc =1.0-np.not_equal(gold_ys, majority_ys).sum()*1.0/len(gold_ys)
                test_acc_both=1.0-error_both/(len(test_batch_start))
                if test_acc_1 > max_acc_test:
                    max_acc_test=test_acc_1
                if test_acc_2 > max_acc_test:
                    max_acc_test=test_acc_2
                if test_acc_3 > max_acc_test:
                    max_acc_test=test_acc_3
                if test_acc_4 > max_acc_test:
                    max_acc_test=test_acc_4
                if test_acc_5 > max_acc_test:
                    max_acc_test=test_acc_5

                if test_acc_both > max_acc_test:
                    max_acc_test=test_acc_both
                    store_model_to_file('/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/model_para_five_copies_'+str(max_acc_test), params)
                if majority_acc> max_majority_acc:
                    max_majority_acc=majority_acc
                    
                print '\t\tcurrent acc:', test_acc_1,test_acc_2,test_acc_3,test_acc_4,test_acc_5,' ; ',test_acc_both,majority_acc, '\t\t\t\t\tmax_acc:', max_acc_test,max_majority_acc
#                 else:
#                     print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev



        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return max_acc_test



if __name__ == '__main__':
    evaluate_lenet5()
    #def evaluate_lenet5(learning_rate=0.01, n_epochs=4, L2_weight=0.0000001, div_weight=0.00001, emb_size=300, batch_size=50, filter_size=[3,1], maxSentLen=50, hidden_size=[300,300], margin =0.2, comment='HL relu'):

#     lr_list=[0.01,0.02,0.008,0.005]
#     batch_list=[3,5,10,20,30,40,50,60,70,80,100]
#     maxlen_list=[35,40,45,50,55,60,65,70,75,80]
#
#     best_acc=0.0
#     best_lr=0.01
#     for lr in lr_list:
#         acc_test= evaluate_lenet5(learning_rate=lr)
#         if acc_test>best_acc:
#             best_lr=lr
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     best_batch=50
#     for batch in batch_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  batch_size=batch)
#         if acc_test>best_acc:
#             best_batch=batch
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#
#     best_maxlen=50
#     for maxlen in maxlen_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  batch_size=best_batch, maxSentLen=maxlen)
#         if acc_test>best_acc:
#             best_maxlen=maxlen
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' batch: ', best_batch, ' maxlen: ', best_maxlen