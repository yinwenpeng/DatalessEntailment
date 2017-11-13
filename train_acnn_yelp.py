import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy as np
import theano
import theano.tensor as T
import random
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from logistic_sgd import LogisticRegression

from theano.tensor.signal import downsample
from random import shuffle
from mlp import HiddenLayer

from load_data import load_yelp_dataset, load_word2vec_file, load_word2vec_to_init
from common_functions import Conv_with_Mask_with_Gate,Conv_for_Pair_SoftAttend,dropout_layer,normalize_matrix,create_conv_para, Conv_for_Pair,Conv_with_Mask, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, L2norm_paraList, Diversify_Reg, create_HiddenLayer_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
def evaluate_lenet5(learning_rate=0.01, n_epochs=100, L2_weight=0.000001, drop_p=0.05, emb_size=300, hidden_size = 500, HL_hidden_size=500,batch_size=5, filter_size=[3,5,7], maxSentLen=180, comment=''):

    model_options = locals().copy()
    print "model options", model_options

    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results
    srng = RandomStreams(rng.randint(999999))
    all_sentences, all_masks, all_labels, word2id=load_yelp_dataset(maxlen=maxSentLen, minlen=2)  #minlen, include one label, at least one word in the sentence
    train_sents=np.asarray(all_sentences[0], dtype='int32')
    train_masks=np.asarray(all_masks[0], dtype=theano.config.floatX)
    train_labels=np.asarray(all_labels[0], dtype='int32')
    train_size=len(train_labels)

    dev_sents=np.asarray(all_sentences[1], dtype='int32')
    dev_masks=np.asarray(all_masks[1], dtype=theano.config.floatX)
    dev_labels=np.asarray(all_labels[1], dtype='int32')
    dev_size=len(dev_labels)

    test_sents=np.asarray(all_sentences[2], dtype='int32')
    test_masks=np.asarray(all_masks[2], dtype=theano.config.floatX)
    test_labels=np.asarray(all_labels[2], dtype='int32')
    test_size=len(test_labels)

    vocab_size=  len(word2id)+1 # add one zero pad index

    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec_file('glove.840B.300d.txt')#
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    #now, start to build the input form of the model
    sents_id_matrix=T.imatrix('sents_id_matrix')
    sents_mask=T.fmatrix('sents_mask')
    labels=T.ivector('labels')
    train_flag = T.iscalar()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    common_input=embeddings[sents_id_matrix.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM

#     drop_common_input = dropout_layer(srng, common_input, drop_p, train_flag)


    bow = T.sum(common_input*sents_mask.dimshuffle(0,'x',1), axis=2) #(batch, emb_size)

    gate_filter_shape=(emb_size, 1, emb_size, 1)
    conv_W_2_pre, conv_b_2_pre=create_conv_para(rng, filter_shape=gate_filter_shape)
    conv_W_2_gate, conv_b_2_gate=create_conv_para(rng, filter_shape=gate_filter_shape)

    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size[0]))
    conv_W_context, conv_b_context=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, 1))

    conv_W2, conv_b2=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size[1]))
    conv_W2_context, conv_b2_context=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, 1))

#     conv_W3, conv_b3=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size[2]))
#     conv_W3_context, conv_b3_context=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, 1))
    # conv_W_into_matrix=conv_W.reshape((conv_W.shape[0], conv_W.shape[2]*conv_W.shape[3]))
    soft_att_W_big, soft_att_b_big = create_HiddenLayer_para(rng, emb_size*2, emb_size)
    soft_att_W_small, _ = create_HiddenLayer_para(rng, emb_size, 1)
    soft_att_W2_big, soft_att_b2_big = create_HiddenLayer_para(rng, emb_size*2, emb_size)
    soft_att_W2_small, _ = create_HiddenLayer_para(rng, emb_size, 1)

#     soft_att_W3_big, soft_att_b3_big = create_HiddenLayer_para(rng, emb_size*2, emb_size)
#     soft_att_W3_small, _ = create_HiddenLayer_para(rng, emb_size, 1)

    NN_para=[conv_W_2_pre, conv_b_2_pre,conv_W_2_gate, conv_b_2_gate,
             conv_W, conv_b,conv_W_context,
             conv_W2, conv_b2,conv_W2_context,
#              conv_W3, conv_b3,conv_W3_context,
             soft_att_W_big, soft_att_b_big,soft_att_W_small,
             soft_att_W2_big, soft_att_b2_big,soft_att_W2_small
#              soft_att_W3_big, soft_att_b3_big,soft_att_W3_small
             ]#,conv_W3, conv_b3,conv_W3_context]

    conv_layer_1_gate_l = Conv_with_Mask_with_Gate(rng, input_tensor3=common_input,
             mask_matrix = sents_mask,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             filter_shape=gate_filter_shape,
             W=conv_W_2_pre, b=conv_b_2_pre,
             W_gate =conv_W_2_gate, b_gate=conv_b_2_gate )

    advanced_sent_tensor3 = conv_layer_1_gate_l.output_tensor3

#     conv_layer_pair = Conv_for_Pair(rng,
#             origin_input_tensor3=advanced_sent_tensor3,
#             origin_input_tensor3_r = advanced_sent_tensor3,
#             input_tensor3=advanced_sent_tensor3,
#             input_tensor3_r = advanced_sent_tensor3,
#              mask_matrix = sents_mask,
#              mask_matrix_r = sents_mask,
#              image_shape=(batch_size, 1, emb_size, maxSentLen),
#              image_shape_r = (batch_size, 1, emb_size, maxSentLen),
#              filter_shape=(hidden_size, 1, emb_size, filter_size[0]),
#              filter_shape_context=(hidden_size, 1, emb_size, 1),
#              W=conv_W, b=conv_b,
#              W_context=conv_W_context, b_context=conv_b_context)

    conv_layer_pair = Conv_for_Pair_SoftAttend(rng,
                                               origin_input_tensor3=advanced_sent_tensor3,
                                               origin_input_tensor3_r=advanced_sent_tensor3,
                                               input_tensor3=advanced_sent_tensor3,
                                               input_tensor3_r=advanced_sent_tensor3,
                                               mask_matrix=sents_mask,
                                               mask_matrix_r=sents_mask,
                                               filter_shape=(hidden_size, 1, emb_size, filter_size[0]),
                                               filter_shape_context=(hidden_size, 1, emb_size, 1),
                                               image_shape=(batch_size, 1, emb_size, maxSentLen),
                                               image_shape_r= (batch_size, 1, emb_size, maxSentLen),
                                               W=conv_W, b=conv_b,
                                               W_context=conv_W_context, b_context=conv_b_context,
                                               soft_att_W_big=soft_att_W_big, soft_att_b_big=soft_att_b_big,
                                               soft_att_W_small=soft_att_W_small)




#     conv_layer_2_pair = Conv_for_Pair(rng,
#             origin_input_tensor3=advanced_sent_tensor3,
#             origin_input_tensor3_r = advanced_sent_tensor3,
#             input_tensor3=advanced_sent_tensor3,
#             input_tensor3_r = advanced_sent_tensor3,
#              mask_matrix = sents_mask,
#              mask_matrix_r = sents_mask,
#              image_shape=(batch_size, 1, emb_size, maxSentLen),
#              image_shape_r = (batch_size, 1, emb_size, maxSentLen),
#              filter_shape=(hidden_size, 1, emb_size, filter_size[1]),
#              filter_shape_context=(hidden_size, 1, emb_size, 1),
#              W=conv_W2, b=conv_b2,
#              W_context=conv_W2_context, b_context=conv_b2_context)
    conv_layer_2_pair = Conv_for_Pair_SoftAttend(rng,
                                               origin_input_tensor3=advanced_sent_tensor3,
                                               origin_input_tensor3_r=advanced_sent_tensor3,
                                               input_tensor3=advanced_sent_tensor3,
                                               input_tensor3_r=advanced_sent_tensor3,
                                               mask_matrix=sents_mask,
                                               mask_matrix_r=sents_mask,
                                               filter_shape=(hidden_size, 1, emb_size, filter_size[1]),
                                               filter_shape_context=(hidden_size, 1, emb_size, 1),
                                               image_shape=(batch_size, 1, emb_size, maxSentLen),
                                               image_shape_r= (batch_size, 1, emb_size, maxSentLen),
                                               W=conv_W2, b=conv_b2,
                                               W_context=conv_W2_context, b_context=conv_b2_context,
                                               soft_att_W_big=soft_att_W2_big, soft_att_b_big=soft_att_b2_big,
                                               soft_att_W_small=soft_att_W2_small)

#     conv_layer_3_pair = Conv_for_Pair_SoftAttend(rng,
#                                                origin_input_tensor3=advanced_sent_tensor3,
#                                                origin_input_tensor3_r=advanced_sent_tensor3,
#                                                input_tensor3=advanced_sent_tensor3,
#                                                input_tensor3_r=advanced_sent_tensor3,
#                                                mask_matrix=sents_mask,
#                                                mask_matrix_r=sents_mask,
#                                                filter_shape=(hidden_size, 1, emb_size, filter_size[2]),
#                                                filter_shape_context=(hidden_size, 1, emb_size, 1),
#                                                image_shape=(batch_size, 1, emb_size, maxSentLen),
#                                                image_shape_r= (batch_size, 1, emb_size, maxSentLen),
#                                                W=conv_W3, b=conv_b3,
#                                                W_context=conv_W3_context, b_context=conv_b3_context,
#                                                soft_att_W_big=soft_att_W3_big, soft_att_b_big=soft_att_b3_big,
#                                                soft_att_W_small=soft_att_W3_small)

    # biased_sent_embeddings = conv_layer_pair.biased_attentive_maxpool_vec_l
    sent_embeddings = conv_layer_pair.maxpool_vec_l
    att_sent_embeddings = conv_layer_pair.attentive_maxpool_vec_l


    sent_embeddings_2 = conv_layer_2_pair.maxpool_vec_l
    att_sent_embeddings_2 = conv_layer_2_pair.attentive_maxpool_vec_l




    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    HL_input = T.concatenate([bow,
                              sent_embeddings, att_sent_embeddings,
                              sent_embeddings_2, att_sent_embeddings_2
#                               sent_embeddings_3,att_sent_embeddings_3,
                              ], axis=1)
    HL_input_size = hidden_size*4+emb_size

    HL_layer_1_W, HL_layer_1_b = create_HiddenLayer_para(rng, HL_input_size, HL_hidden_size)
    HL_layer_1_params = [HL_layer_1_W, HL_layer_1_b]
    HL_layer_1=HiddenLayer(rng, input=HL_input, n_in=HL_input_size, n_out=HL_hidden_size, W=HL_layer_1_W, b=HL_layer_1_b, activation=T.nnet.relu)
#     HL_layer_1_output = dropout_layer(srng, HL_layer_1.output, drop_p, train_flag)

    HL_layer_2_W, HL_layer_2_b = create_HiddenLayer_para(rng, HL_hidden_size, HL_hidden_size)
    HL_layer_2_params = [HL_layer_2_W, HL_layer_2_b]
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=HL_hidden_size, n_out=HL_hidden_size, W=HL_layer_2_W, b=HL_layer_2_b, activation=T.nnet.relu)
#     HL_layer_2_output = dropout_layer(srng, HL_layer_2.output, drop_p, train_flag)

    LR_input = T.concatenate([HL_input, HL_layer_1.output, HL_layer_2.output], axis=1)
#     drop_LR_input = dropout_layer(srng, LR_input, drop_p, train_flag)
    LR_input_size = HL_input_size+2*HL_hidden_size


    U_a = create_ensemble_para(rng, 5, LR_input_size) # the weight matrix hidden_size*2
#     norm_W_a = normalize_matrix(U_a)
    LR_b = theano.shared(value=np.zeros((5,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]
    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=5, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.

    params = [embeddings]+NN_para+HL_layer_1_params+HL_layer_2_params+LR_para   # put all model parameters together
    L2_reg =L2norm_paraList([embeddings,
                             conv_W_2_pre,conv_W_2_gate,conv_W,conv_W_context,conv_W2,conv_W2_context,
                             soft_att_W_big,soft_att_W_small,
                             soft_att_W2_big,soft_att_W2_small,
                             HL_layer_1_W,HL_layer_2_W,U_a])
#     diversify_reg= Diversify_Reg(U_a.T)+Diversify_Reg(conv_W_into_matrix)

    cost=loss#+L2_weight*L2_reg

    grads = T.grad(cost, params)    # create a list of gradients for all model parameters

    accumulator=[]
    for para_i in params:
        eps_p=np.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #1e-8 is add to get rid of zero division
        updates.append((acc_i, acc))

    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([sents_id_matrix, sents_mask, labels,train_flag], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([sents_id_matrix, sents_mask, labels,train_flag], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_id_matrix, sents_mask, labels,train_flag], [layer_LR.errors(labels), layer_LR.y_pred], allow_input_downcast=True, on_unused_input='ignore')

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
    cost_i=0.0
    train_indices = range(train_size)
    while epoch < n_epochs:
        epoch = epoch + 1
        # combined = zip(train_sents, train_masks, train_labels)
        random.Random(200).shuffle(train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0

        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+batch_size]
            cost_i+= train_model(
                                train_sents[train_id_batch],
                                train_masks[train_id_batch],
                                train_labels[train_id_batch],
                                1)

            #after each 1000 batches, we test the performance of the model on all test data
            if iter %2000==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()


                error_sum=0.0
                # writefile=open('log.'+nn+'.senti.preditions.txt', 'w')
                for test_batch_id in test_batch_start: # for each test batch
                    error_i, pred_labels=test_model(
                                                    test_sents[test_batch_id:test_batch_id+batch_size],
                                                    test_masks[test_batch_id:test_batch_id+batch_size],
                                                    test_labels[test_batch_id:test_batch_id+batch_size],
                                                    0)
                    # pred_labels=list(pred_labels)
                    # if test_batch_id !=test_batch_start[-1]:
                    #     writefile.write('\n'.join(map(str,pred_labels))+'\n')
                    # else:
                    #     writefile.write('\n'.join(map(str,pred_labels[-test_size%batch_size:])))

                    error_sum+=error_i
                # writefile.close()
                test_accuracy=1.0-error_sum/(len(test_batch_start))
                if test_accuracy > max_acc_test:
                    max_acc_test=test_accuracy
                print '\t\tcurrent testbacc:', test_accuracy, '\t\t\t\t\tmax_acc_test:', max_acc_test


        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return max_acc_test



if __name__ == '__main__':
#     evaluate_lenet5(learning_rate=0.2, emb_size=20, batch_size=50, maxSentLen=60, filter_size=3, nn='CNN')
#     evaluate_lenet5(learning_rate=0.1, emb_size=30, batch_size=50, maxSentLen=60, nn='GRU')
    evaluate_lenet5()
    #(learning_rate=0.1, n_epochs=2000, L2_weight=0.001, emb_size=13, batch_size=50, filter_size=3, maxSentLen=60)
#     lr_list=[0.1,0.05,0.01,0.005,0.001,0.2,0.3,0.4,0.5]
#     emb_list=[5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,150,200,250,300]
#     batch_list=[5,10,20,30,40,50,60,70,80,100]
#     maxlen_list=[5,10,15,20,25,30,35,40,45,50,55,60,65,70]
#
#     best_acc=0.0
#     best_lr=0.1
#     for lr in lr_list:
#         acc_test= evaluate_lenet5(learning_rate=lr)
#         if acc_test>best_acc:
#             best_lr=lr
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#
#     best_emb=13
#     for emb in emb_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr, emb_size=emb)
#         if acc_test>best_acc:
#             best_emb=emb
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#
#     best_batch=50
#     for batch in batch_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=batch)
#         if acc_test>best_acc:
#             best_batch=batch
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#
#     best_maxlen=60
#     for maxlen in maxlen_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=best_batch, maxSentLen=maxlen)
#         if acc_test>best_acc:
#             best_maxlen=maxlen
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' emb: ', best_emb, ' batch: ', best_batch, ' maxlen: ', best_maxlen
