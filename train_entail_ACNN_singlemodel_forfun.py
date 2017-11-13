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

from load_data import load_SNLI_dataset_with_extra, load_word2vec, load_word2vec_to_init, extend_word2vec_lowercase,load_SNLI_dataset_with_extra_with_test
from common_functions import one_classifier_in_one_copy,Adam,create_HiddenLayer_para,dropit,create_ensemble_para_with_bounds, Conv_for_Pair,dropout_layer, store_model_to_file, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, ABCNN, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
'''
1, use SVM outside
drop0.05, reach 0.86345177665
2, use local context, self-global context, context from counterpart for CNN
3, context from counterpart and self-global consider distance bias
4, check output labels, if three group has the same predictions, use another score to refine them
'''

def evaluate_lenet5(learning_rate=0.02, n_epochs=4, L2_weight=0.0000001, extra_size=4, drop_p=0.2, div_weight=0.00001, emb_size=300,
                    batch_size=50, filter_size=[3,3], maxSentLen=40, hidden_size=[300,300], comment=''):

    model_options = locals().copy()
    print "model options", model_options

    first_seeds=[1234,1235,1236,1237] #first copy starts by 1
    first_rngs = [np.random.RandomState(first_seeds[0]),np.random.RandomState(first_seeds[1]),np.random.RandomState(first_seeds[2]),np.random.RandomState(first_seeds[3])]    #random seed, control the model generates the same results
    first_srng = RandomStreams(first_rngs[0].randint(999999))


    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_extra, all_labels, word2id, test_rows  =load_SNLI_dataset_with_extra_with_test(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
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


    rand_values=first_rngs[0].normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
#     word2vec =extend_word2vec_lowercase(word2vec)
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    first_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    #now, start to build the input form of the model
    train_flag = T.iscalar()
    first_sents_ids_l=T.imatrix()
    first_sents_mask_l=T.fmatrix()
    first_sents_ids_r=T.imatrix()
    first_sents_mask_r=T.fmatrix()
    first_labels=T.ivector()

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    def common_input(emb_matrix, sent_ids):
        return emb_matrix[sent_ids.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    first_common_input_l=dropout_layer(first_srng, common_input(first_embeddings, first_sents_ids_l), drop_p, train_flag)#embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    first_common_input_r=dropout_layer(first_srng, common_input(first_embeddings, first_sents_ids_r), drop_p, train_flag)#embeddings[sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    gate_filter_shape=(hidden_size[0], 1, emb_size, 1)
    def create_CNN_params(rng):
        conv_W_2_pre, conv_b_2_pre=create_conv_para(rng, filter_shape=gate_filter_shape)
        conv_W_2_gate, conv_b_2_gate=create_conv_para(rng, filter_shape=gate_filter_shape)
        conv_W_2, conv_b_2=create_conv_para(rng, filter_shape=(hidden_size[1], 1, hidden_size[0], filter_size[0]))
        conv_W_2_context, conv_b_2_context=create_conv_para(rng, filter_shape=(hidden_size[1], 1, hidden_size[0], 1))
        return conv_W_2_pre, conv_b_2_pre,conv_W_2_gate, conv_b_2_gate,conv_W_2, conv_b_2,conv_W_2_context, conv_b_2_context

    first_conv_W_pre, first_conv_b_pre,first_conv_W_gate, first_conv_b_gate,first_conv_W, first_conv_b,first_conv_W_context, first_conv_b_context = create_CNN_params(first_rngs[0])


    first_CNN_1_para=[
            first_conv_W_pre, first_conv_b_pre,
            first_conv_W_gate, first_conv_b_gate,
            first_conv_W, first_conv_b,first_conv_W_context]



    '''
    first copy
    '''
    def copy(rngs,common_input_l,common_input_r,sents_mask_l, sents_mask_r,
                   drop_conv_W_1_pre,conv_b_1_pre,drop_conv_W_1_gate,conv_b_1_gate,
                   drop_conv_W_1,conv_b_1,drop_conv_W_1_context, conv_b_1_context,
                   labels):

        loss_0_0, distr_0_0, params_0_0 = one_classifier_in_one_copy(rngs[0], common_input_l,common_input_r,sents_mask_l,sents_mask_r,batch_size, emb_size,
                                                                     maxSentLen,gate_filter_shape,hidden_size,filter_size,
                               first_srng, drop_p,train_flag,labels,
                               drop_conv_W_1_pre,conv_b_1_pre,drop_conv_W_1_gate,conv_b_1_gate,
                               drop_conv_W_1,conv_b_1,drop_conv_W_1_context,conv_b_1_context,
                               True)
        # loss_0_4, distr_0_4, params_0_4 = one_classifier_in_one_copy(rng, common_input_l,common_input_r,sents_mask_l,sents_mask_r,batch_size, emb_size,
        #                                                              maxSentLen,gate_filter_shape,hidden_size,filter_size,
        #                        first_srng, drop_p,train_flag,labels,
        #                        drop_conv_W_1_pre_5,conv_b_1_pre_5,drop_conv_W_1_gate_5,conv_b_1_gate_5,
        #                        drop_conv_W_1_5,conv_b_1_5,drop_conv_W_1_context_5,conv_b_1_context_5,
        #                        True)
        # loss_0_5, distr_0_5, params_0_5 = one_classifier_in_one_copy(rng, common_input_l,common_input_r,sents_mask_l,sents_mask_r,batch_size, emb_size,
        #                                                              maxSentLen,gate_filter_shape,hidden_size,filter_size,
        #                        first_srng, drop_p,train_flag,labels,
        #                        drop_conv_W_1_pre_6,conv_b_1_pre_6,drop_conv_W_1_gate_6,conv_b_1_gate_6,
        #                        drop_conv_W_1_6,conv_b_1_6,drop_conv_W_1_context_6,conv_b_1_context_6,
        #                        False)

#         psp_label = T.repeat(labels, multi_psp_size)

        loss_0=loss_0_0
        para_0 = params_0_0


#         loss = loss_0+loss_1+loss_2
        batch_distr = distr_0_0#T.sum((layer_LR.p_y_given_x).reshape((batch_size, multi_psp_size,3)), axis=1)  #(batch, 3)

        return loss_0, para_0, batch_distr

    first_loss, first_classifier_params, first_test_distr = copy(first_rngs,first_common_input_l,first_common_input_r,first_sents_mask_l,first_sents_mask_r,
                                                                                    first_conv_W_pre,first_conv_b_pre,first_conv_W_gate,first_conv_b_gate,
                                                                                    first_conv_W,first_conv_b,first_conv_W_context,first_conv_b_context,
                                                                                    first_labels)



    first_preds = T.argmax(first_test_distr, axis=1) #batch
    all_error = T.mean(T.neq(first_preds, first_labels))


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
    
    first_common_para = [first_embeddings]
    first_classifier_1_para = first_CNN_1_para+first_classifier_params
    
    first_common_updates =   Gradient_Cost_Para(first_loss,first_common_para, learning_rate)
    first_classifier_1_updates =   Gradient_Cost_Para(first_loss,first_classifier_1_para, learning_rate)


    cost=first_loss


    first_updates =   first_common_updates+first_classifier_1_updates

    updates = first_updates

    #train_model = theano.function([sents_id_matrix, sents_mask, labels], cost, updates=updates, on_unused_input='ignore')
    train_model = theano.function([train_flag,first_sents_ids_l, first_sents_mask_l, first_sents_ids_r, first_sents_mask_r, first_labels,
                                   ], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
#     train_model_pred = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,extra,labels,
#                                         second_sents_ids_l,second_sents_mask_l,second_sents_ids_r,second_sents_mask_r,second_labels], [LR_input, labels], allow_input_downcast=True, on_unused_input='ignore')
#
#     dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, train_flag,extra, labels,
#                                  second_sents_ids_l,second_sents_mask_l,second_sents_ids_r,second_sents_mask_r,second_labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([train_flag,first_sents_ids_l, first_sents_mask_l, first_sents_ids_r, first_sents_mask_r, first_labels,
                                  ], [all_error,first_preds], allow_input_downcast=True, on_unused_input='ignore')

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
    gold_test_rows = test_rows[:(n_test_batches*batch_size)]+test_rows[-batch_size:]


    max_acc_test=0.0


    cost_i=0.0
    first_train_indices = range(train_size)
    while epoch < n_epochs:
        epoch = epoch + 1

        random.Random(200).shuffle(first_train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0

        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            first_train_id_batch = first_train_indices[batch_id:batch_id+batch_size]
            cost_i+= train_model(
                                1,
                                train_sents_l[first_train_id_batch],
                                train_masks_l[first_train_id_batch],
                                train_sents_r[first_train_id_batch],
                                train_masks_r[first_train_id_batch],
                                train_labels_store[first_train_id_batch])

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
#                 error_sum_1=0.0
#                 error_sum_2=0.0
                error_sum_comb=0.0
                pred_ys = []
                for test_batch_id in test_batch_start: # for each test batch
                    error_comb, pred_ys_batch=test_model(
                            0,
                            test_sents_l[test_batch_id:test_batch_id+batch_size],
                            test_masks_l[test_batch_id:test_batch_id+batch_size],
                            test_sents_r[test_batch_id:test_batch_id+batch_size],
                            test_masks_r[test_batch_id:test_batch_id+batch_size],
                            test_labels_store[test_batch_id:test_batch_id+batch_size]
                            )

#                     error_sum_1+=error_1
#                     error_sum_2+=error_2
                    error_sum_comb+=error_comb
                    pred_ys+=list(pred_ys_batch)

#                 test_acc_1=1.0-error_sum_1/(len(test_batch_start))
#                 test_acc_2=1.0-error_sum_2/(len(test_batch_start))
                test_acc_comb=1.0-error_sum_comb/(len(test_batch_start))

#                 if test_acc_1 > max_acc_test:
#                     max_acc_test=test_acc_1
#                 if test_acc_2 > max_acc_test:
#                     max_acc_test=test_acc_2
                if test_acc_comb > max_acc_test:
                    max_acc_test=test_acc_comb
#                     store_model_to_file('/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/model_para_single_model_'+str(max_acc_test), params)

                    if len(pred_ys)!=len(gold_test_rows):
                        print 'len(pred_ys)!=len(gold_test_rows):', len(pred_ys), len(gold_test_rows)
                    else:
                        test_write=open('/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/error_analysis_'+str(max_acc_test)+'.txt', 'w')
                        for i in range(len(pred_ys)):
                            test_write.write(str(pred_ys[i])+'\t'+gold_test_rows[i]+'\n')
                        print 'error analysis file written over.'
                        test_write.close()


                print '\t\tcurrent acc:', test_acc_comb, '\t\t\t\t\tmax_acc:', max_acc_test
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
