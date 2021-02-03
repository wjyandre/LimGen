"""
Author:         Jey Han Lau
Date:           Aug 17

Modified by Jianyou (Andre) Wang for limerick generation.
Removed pentameter and rhyming model, character level LSTM
since available limerick training dataset (<2k) is insufficient.
Added rule-based methods to enforce rhyming, meter , syllable
constraints, similar to LimGen.
Date: Sep 2020
"""

import argparse
import sys
import codecs
import random
import time
import imp
import os
import cPickle
import tensorflow as tf
import numpy as np
import gensim.models as g
from util import *
from sonnet_model import SonnetModel
from sklearn.metrics import roc_auc_score
from nltk.corpus import cmudict
import pdb
#parser arguments
desc = "trains sonnet model"
parser = argparse.ArgumentParser(description=desc)

#arguments
parser.add_argument("-c", "--config", help="path of config file")
args = parser.parse_args()

#constants
pad_symbol = "<pad>"
end_symbol = "<eos>"
unk_symbol = "<unk>"
dummy_symbols = [pad_symbol, end_symbol, unk_symbol]

#globals
wordxid = None
idxword = None
charxid = None
idxchar = None
wordxchar = None #word id to [char ids]
rhyme_thresholds = [0.9, 0.8, 0.7, 0.6]
stress_acc_threshold = 0.4
reset_scale = 1.05

#training parameters
train_lm = True
train_pm = True
train_rm = True


###########
#functions#
###########

def run_epoch(sess, word_batches, model, pname, is_training):

    start_time  = time.time()

    #lm variables
    lm_costs    = 0.0
    total_words = 0
    zero_state  = sess.run(model.lm_initial_state)
    model_state = None
    prev_doc    = -1
    lm_train_op = model.lm_train_op if is_training else tf.no_op()

    #mix lm and pm batches
    mixed_batch_types = [0]*len(word_batches)
    random.shuffle(mixed_batch_types)
    mixed_batches = [word_batches]#, char_batches, rhyme_batches]

    word_batch_id  = 0
    char_batch_id  = 0
    rhyme_batch_id = 0


    for bi, batch_type in enumerate(mixed_batch_types):

        if batch_type == 0 and train_lm:
    
            b = mixed_batches[batch_type][word_batch_id]

            #reset model state if it's a different set of documents
            if prev_doc != b[2][0]: 
                model_state = zero_state
                prev_doc = b[2][0]

            #preprocess character input to [batch_size*doc_len, char_len]
            pm_enc_x = np.array(b[5]).reshape((cf.batch_size*max(b[3]), -1))

            feed_dict = {model.lm_x: b[0], model.lm_y: b[1], model.lm_xlen: b[3], model.pm_enc_x: pm_enc_x,
                model.pm_enc_xlen: np.array(b[6]).reshape((-1)), model.lm_initial_state: model_state,
                model.lm_hist: b[7], model.lm_hlen: b[8]}

            cost, model_state, attns, _ = sess.run([model.lm_cost, model.lm_final_state, model.lm_attentions, lm_train_op],
                feed_dict)

            lm_costs    += cost * cf.batch_size #keep track of full cost
            total_words += sum(b[3])

            word_batch_id += 1
        '''
        elif batch_type == 1 and train_pm:
    
            b = mixed_batches[batch_type][char_batch_id]

            feed_dict        = {model.pm_enc_x: b[0], model.pm_enc_xlen: b[1], model.pm_cov_mask: b[2]}
            cost, attns, _,  = sess.run([model.pm_mean_cost, model.pm_attentions, pm_train_op], feed_dict)
            pm_costs        += cost
            
            char_batch_id += 1

            if not is_training:
                eval_stress(stress_acc, cmu, attns, model.pentameter, b[0], idxchar, charxid, pad_symbol, cf)

        '''
        if (((bi % 10) == 0) and cf.verbose) or (bi == len(mixed_batch_types)-1):

            partition = "  " + pname
            sent_end  = "\n" if bi == (len(mixed_batch_types)-1) else "\r"
            speed     = (bi+1)/(time.time()-start_time)

            sys.stdout.write("%s %d/%d: lm ppl = %.1f; batch/sec = %.1f%s" % \
                (partition, bi+1, len(mixed_batch_types), np.exp(lm_costs/max(total_words, 1)), speed, sent_end))
            sys.stdout.flush()
        '''
            if not is_training and (bi == len(mixed_batch_types)-1):

                if train_pm:
                    all_acc = [ item for sublist in stress_acc for item in sublist ]
                    stress_acc.append(all_acc)
                    for acci, acc in enumerate(stress_acc):
                        sys.stdout.write("    Stress acc [%d]   = %.3f (%d)\n" % (acci, np.mean(acc), len(acc)))

                if train_rm:
                    for t in rhyme_thresholds:
                        p = np.mean(rhyme_pr[t][0])
                        r = np.mean(rhyme_pr[t][1])
                        f = 2*p*r / (p+r) if (p != 0.0 and r != 0.0) else 0.0
                        sys.stdout.write("    Rhyme P/R/F@%.1f  = %.3f / %.3f / %.3f\n" % (t, p, r, f))

                sys.stdout.flush()
        '''
    #return avg batch loss for lm, pm and rm
    return lm_costs/max(word_batch_id, 1)



######
#main#
######

def main():

    global wordxid, idxword, charxid, idxchar, wordxchar, train_lm, train_pm, train_rm

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

    #set the seeds
    random.seed(cf.seed)
    np.random.seed(cf.seed)

    #load word embedding model if given and set word embedding size
    if cf.word_embedding_model:
        print "\nLoading word embedding model..."
        mword = g.Word2Vec.load(cf.word_embedding_model)
        cf.word_embedding_dim= mword.vector_size

    #load vocab
    print "\nFirst pass to collect word and character vocabulary..."
    idxword, wordxid, idxchar, charxid, wordxchar = load_vocab(cf.train_data, cf.word_minfreq, dummy_symbols)
    print "\nWord type size =", len(idxword)
    print "\nChar type size =", len(idxchar)

    #load train and valid data
    print "\nLoading train and valid data..."
    train_word_data, train_char_data, train_rhyme_data, train_nwords, train_nchars = \
        load_data(cf.train_data, wordxid, idxword, charxid, idxchar, dummy_symbols)
    valid_word_data, valid_char_data, valid_rhyme_data, valid_nwords, valid_nchars = \
        load_data(cf.valid_data, wordxid, idxword, charxid, idxchar, dummy_symbols)
    print_stats("\nTrain", train_word_data, train_rhyme_data, train_nwords, train_nchars)
    print_stats("\nValid", valid_word_data, valid_rhyme_data, valid_nwords, valid_nchars)


    #train model
    with tf.Graph().as_default(), tf.Session() as sess:

        tf.set_random_seed(cf.seed)

        with tf.variable_scope("model", reuse=None):
            mtrain = SonnetModel(True, cf.batch_size, len(idxword), len(idxchar),
                charxid[" "], charxid[pad_symbol], cf)
        with tf.variable_scope("model", reuse=True):
            mvalid = SonnetModel(False, cf.batch_size, len(idxword), len(idxchar),
                charxid[" "], charxid[pad_symbol], cf)
        with tf.variable_scope("model", reuse=True):
            mgen = SonnetModel(False, 1, len(idxword), len(idxchar), charxid[" "], charxid[pad_symbol], cf)

        tf.global_variables_initializer().run()

        #initialise word embedding
        if cf.word_embedding_model:
            word_emb = init_embedding(mword, idxword)
            sess.run(mtrain.word_embedding.assign(word_emb))

        #save model
        if cf.save_model:
            if not os.path.exists(os.path.join(cf.output_dir, "trained_model")):
                os.makedirs(os.path.join(cf.output_dir, "trained_model"))
            #create saver object to save model
            saver = tf.train.Saver(max_to_keep=0)

        #train model
        prev_lm_loss= None
        for i in xrange(cf.epoch_size):

            print "\nEpoch =", i+1

            #create batches for language model
            train_word_batch = create_word_batch(train_word_data, cf.batch_size,
                cf.doc_lines, cf.bptt_truncate, wordxid[pad_symbol], wordxid[end_symbol], wordxid[unk_symbol], True)
            valid_word_batch = create_word_batch(valid_word_data, cf.batch_size,
                cf.doc_lines, cf.bptt_truncate, wordxid[pad_symbol], wordxid[end_symbol], wordxid[unk_symbol], False)
            '''
            #create batches for pentameter model
            train_char_batch = create_char_batch(train_char_data, cf.batch_size,
                charxid[pad_symbol], mtrain.pentameter, idxchar, True)
            valid_char_batch = create_char_batch(valid_char_data, cf.batch_size,
                charxid[pad_symbol], mtrain.pentameter, idxchar, False)

            #create batches for rhyme model
            train_rhyme_batch = create_rhyme_batch(train_rhyme_data, cf.batch_size, charxid[pad_symbol], wordxchar,
                cf.rm_neg, True)
            valid_rhyme_batch = create_rhyme_batch(valid_rhyme_data, cf.batch_size, charxid[pad_symbol], wordxchar,
                cf.rm_neg, False)
            '''
            #train an epoch
            _ = run_epoch(sess, train_word_batch, mtrain, "TRAIN", True)
            lm_loss = run_epoch(sess, valid_word_batch, mvalid, "VALID", False)


            #save model
            if cf.save_model:
                if prev_lm_loss == None  or \
                    (lm_loss <= prev_lm_loss or not train_lm):
                    saver.save(sess, os.path.join(cf.output_dir, "trained_model", "model.ckpt"))
                    prev_lm_loss = lm_loss
                else:
                    saver.restore(sess, os.path.join(cf.output_dir, "trained_model", "model.ckpt"))
                    print "New valid performance is worse; restoring previous parameters..."
                    print "  lm loss: %.5f --> %.5f" % (prev_lm_loss, lm_loss)
                    sys.stdout.flush()


        #save vocab information and config
        if cf.save_model:
            #vocab
            cPickle.dump((idxword, idxchar, wordxchar), \
                open(os.path.join(cf.output_dir, "trained_model", "vocabs.pickle"), "w"))

            #create a dictionary object for config
            cf_dict = {}
            for k,v in vars(cf).items():
                if not k.startswith("__"):
                    cf_dict[k] = v
            cPickle.dump(cf_dict, open(os.path.join(cf.output_dir, "trained_model", "config.pickle"), "w"))


if __name__ == "__main__":

    #load config
    if args.config:
        print "Loading config from:", args.config
        cf = imp.load_source('config', args.config)
    else:
        print "Loading config from default directory"
        import config as cf

    main()
