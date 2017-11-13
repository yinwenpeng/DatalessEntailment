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

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle

from load_data import load_SNLI_dataset, load_word2vec, load_word2vec_to_init
from common_functions import create_conv_para, Conv_with_input_para, LSTM_Batch_Tensor_Input_with_Mask, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, create_GRU_para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
def evaluate_lenet5(learning_rate=0.1, n_epochs=4, L2_weight=0.001, emb_size=70, batch_size=50, filter_size=3, maxSentLen=50, nn='CNN'):
    hidden_size=emb_size
    model_options = locals().copy()
    print "model options", model_options
    
    rng = np.random.RandomState(1234)    #random seed, control the model generates the same results 


    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, word2id  =load_SNLI_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    train_sents_l=np.asarray(all_sentences_l[0], dtype='int32')
    dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
    test_sents_l=np.asarray(all_sentences_l[2], dtype='int32')
    
    train_masks_l=np.asarray(all_masks_l[0], dtype=theano.config.floatX)
    dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
    test_masks_l=np.asarray(all_masks_l[2], dtype=theano.config.floatX)
    
    train_sents_r=np.asarray(all_sentences_r[0], dtype='int32')
    dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
    test_sents_r=np.asarray(all_sentences_r[2] , dtype='int32')
        
    train_masks_r=np.asarray(all_masks_r[0], dtype=theano.config.floatX)
    dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
    test_masks_r=np.asarray(all_masks_r[2], dtype=theano.config.floatX)
            
    train_labels_store=np.asarray(all_labels[0], dtype='int32')
    dev_labels_store=np.asarray(all_labels[1], dtype='int32')
    test_labels_store=np.asarray(all_labels[2], dtype='int32')
        
    train_size=len(train_labels_store)
    dev_size=len(dev_labels_store)
    test_size=len(test_labels_store)
    
    vocab_size=len(word2id)+1
                    
    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
#     rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
#     id2word = {y:x for x,y in word2id.iteritems()}
#     word2vec=load_word2vec()
#     rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable      
    
    
    #now, start to build the input form of the model
    sents_ids_l=T.imatrix()
    sents_mask_l=T.fmatrix()
    sents_ids_r=T.imatrix()
    sents_mask_r=T.fmatrix()
    labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'    
    
    common_input_l=embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    common_input_r=embeddings[sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size))
    
    #conv
    if nn=='CNN':
        conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size))
        conv_W_into_matrix=conv_W.reshape((conv_W.shape[0], conv_W.shape[2]*conv_W.shape[3]))
        NN_para=[conv_W, conv_b]
         
        conv_input_l = common_input_l.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_model_l = Conv_with_input_para(rng, input=conv_input_l,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
        conv_output_l=conv_model_l.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3_l=conv_output_l.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
        mask_for_conv_output_l=T.repeat(sents_mask_l[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_l=(1.0-mask_for_conv_output_l)*(mask_for_conv_output_l-10)
        masked_conv_output_l=conv_output_into_tensor3_l+mask_for_conv_output_l      #mutiple mask with the conv_out to set the features by UNK to zero
        sent_embeddings_l=T.max(masked_conv_output_l, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
     
        conv_input_r = common_input_r.dimshuffle((0,'x', 2,1)) #(batch_size, 1, emb_size, maxsenlen)
        conv_model_r = Conv_with_input_para(rng, input=conv_input_r,
                 image_shape=(batch_size, 1, emb_size, maxSentLen),
                 filter_shape=(hidden_size, 1, emb_size, filter_size), W=conv_W, b=conv_b)
        conv_output_r=conv_model_r.narrow_conv_out #(batch, 1, hidden_size, maxsenlen-filter_size+1)    
        conv_output_into_tensor3_r=conv_output_r.reshape((batch_size, hidden_size, maxSentLen-filter_size+1))
        mask_for_conv_output_r=T.repeat(sents_mask_r[:,filter_size-1:].reshape((batch_size, 1, maxSentLen-filter_size+1)), hidden_size, axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output_r=(1.0-mask_for_conv_output_r)*(mask_for_conv_output_r-10)
        masked_conv_output_r=conv_output_into_tensor3_r+mask_for_conv_output_r      #mutiple mask with the conv_out to set the features by UNK to zero
        sent_embeddings_r=T.max(masked_conv_output_r, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size   
    
     
    #GRU
    if nn=='GRU':
        U1, W1, b1=create_GRU_para(rng, emb_size, hidden_size)
        NN_para=[U1, W1, b1]     #U1 includes 3 matrices, W1 also includes 3 matrices b1 is bias
        gru_input_l = common_input_l.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
        gru_layer_l=GRU_Batch_Tensor_Input_with_Mask(gru_input_l, sents_mask_l,  hidden_size, U1, W1, b1)
        sent_embeddings_l=gru_layer_l.output_sent_rep  # (batch_size, hidden_size)
        gru_input_r = common_input_r.dimshuffle((0,2,1))   #gru requires input (batch_size, emb_size, maxSentLen)
        gru_layer_r=GRU_Batch_Tensor_Input_with_Mask(gru_input_r, sents_mask_r,  hidden_size, U1, W1, b1)
        sent_embeddings_r=gru_layer_r.output_sent_rep  # (batch_size, hidden_size)


    #LSTM
    if nn=='LSTM':
        LSTM_para_dict=create_LSTM_para(rng, emb_size, hidden_size)
        NN_para=LSTM_para_dict.values() # .values returns a list of parameters
        lstm_input_l = common_input_l.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
        lstm_layer_l=LSTM_Batch_Tensor_Input_with_Mask(lstm_input_l, sents_mask_l,  hidden_size, LSTM_para_dict)
        sent_embeddings_l=lstm_layer_l.output_sent_rep  # (batch_size, hidden_size)   
        lstm_input_r = common_input_r.dimshuffle((0,2,1)) #LSTM has the same inpur format with GRU
        lstm_layer_r=LSTM_Batch_Tensor_Input_with_Mask(lstm_input_r, sents_mask_r,  hidden_size, LSTM_para_dict)
        sent_embeddings_r=lstm_layer_r.output_sent_rep  # (batch_size, hidden_size)      
    
    
    HL_layer_1_input = T.concatenate([sent_embeddings_l,sent_embeddings_r, sent_embeddings_l*sent_embeddings_r, cosine_matrix1_matrix2_rowwise(sent_embeddings_l,sent_embeddings_r).dimshuffle(0,'x')],axis=1)
    HL_layer_1_input_size = hidden_size*3+1
    HL_layer_1=HiddenLayer(rng, input=HL_layer_1_input, n_in=HL_layer_1_input_size, n_out=hidden_size, activation=T.tanh)
    HL_layer_2=HiddenLayer(rng, input=HL_layer_1.output, n_in=hidden_size, n_out=hidden_size, activation=T.tanh)

    #classification layer, it is just mapping from a feature vector of size "hidden_size" to a vector of only two values: positive, negative
    LR_input_size=HL_layer_1_input_size+2*hidden_size
    U_a = create_ensemble_para(rng, 3, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((3,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class  
    LR_para=[U_a, LR_b]
    
    LR_input=T.concatenate([HL_layer_1_input, HL_layer_1.output, HL_layer_2.output],axis=1)
    layer_LR=LogisticRegression(rng, input=T.tanh(LR_input), n_in=LR_input_size, n_out=3, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.
    
    params = [embeddings]+NN_para+LR_para+HL_layer_1.params+HL_layer_2.params   # put all model parameters together
#     L2_reg =L2norm_paraList([embeddings,conv_W, U_a])
#     diversify_reg= Diversify_Reg(U_a.T)+Diversify_Reg(conv_W_into_matrix)

    cost=loss#+Div_reg*diversify_reg#+L2_weight*L2_reg
    
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
    train_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')    
    test_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')
    
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
    
    while epoch < n_epochs:
        epoch = epoch + 1
        train_indices = range(train_size)
        random.Random(200).shuffle(train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed
        iter_accu=0
        cost_i=0.0
        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+batch_size]
            cost_i+= train_model(
                                train_sents_l[train_id_batch], 
                                train_masks_l[train_id_batch],
                                train_sents_r[train_id_batch], 
                                train_masks_r[train_id_batch],                                
                                train_labels_store[train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%500==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()
#             if epoch >=3 and iter >= len(train_batch_start)*2.0/3 and iter%500==0:
#                 print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
#                 past_time = time.time()

                error_sum=0.0
                for dev_batch_id in dev_batch_start: # for each test batch
                    error_i=dev_model(
                                dev_sents_l[dev_batch_id:dev_batch_id+batch_size], 
                                dev_masks_l[dev_batch_id:dev_batch_id+batch_size],
                                dev_sents_r[dev_batch_id:dev_batch_id+batch_size], 
                                dev_masks_r[dev_batch_id:dev_batch_id+batch_size],                                
                                dev_labels_store[dev_batch_id:dev_batch_id+batch_size]
                                )
                    
                    error_sum+=error_i
                dev_accuracy=1.0-error_sum/(len(dev_batch_start))
                if dev_accuracy > max_acc_dev:
                    max_acc_dev=dev_accuracy
                    print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev
                    #best dev model, do test
                    error_sum=0.0
                    for test_batch_id in test_batch_start: # for each test batch
                        error_i=test_model(
                                test_sents_l[test_batch_id:test_batch_id+batch_size], 
                                test_masks_l[test_batch_id:test_batch_id+batch_size],
                                test_sents_r[test_batch_id:test_batch_id+batch_size], 
                                test_masks_r[test_batch_id:test_batch_id+batch_size],                                
                                test_labels_store[test_batch_id:test_batch_id+batch_size]
                                )
                        
                        error_sum+=error_i
                    test_accuracy=1.0-error_sum/(len(test_batch_start))
                    if test_accuracy > max_acc_test:
                        max_acc_test=test_accuracy
                    print '\t\tcurrent testbacc:', test_accuracy, '\t\t\t\t\tmax_acc_test:', max_acc_test
                else:
                    print 'current dev_accuracy:', dev_accuracy, '\t\t\t\t\tmax max_acc_dev:', max_acc_dev

        
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
    #(learning_rate=0.1, n_epochs=3, L2_weight=0.001, emb_size=30, batch_size=50, filter_size=3, maxSentLen=50, nn='GRU'):
#     lr_list=[0.1,0.05,0.01,0.005,0.001,0.2,0.3]
#     emb_list=[10,20,30,40,50,60,70,80,90,100,120,150,200,250,300]
#     batch_list=[30,40,50,60,70,80,100,150,200,250,300]
#     maxlen_list=[35,40,45,50,55,60,65,70,75,80]
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
#     best_emb=30
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
#     best_maxlen=50        
#     for maxlen in maxlen_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=best_batch, maxSentLen=maxlen)
#         if acc_test>best_acc:
#             best_maxlen=maxlen
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' emb: ', best_emb, ' batch: ', best_batch, ' maxlen: ', best_maxlen
#     
#     
#     
#     
    
    
    
    
    
    
    
    