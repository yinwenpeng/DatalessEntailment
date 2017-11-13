from load_data import load_SNLI_dataset_with_extra, load_word2vec, load_word2vec_file
from difflib import SequenceMatcher
import re
def find_top_similar_vocab(word, vocab, overlap_vocab):
    
    max_ratio=0.0
    max_word=''
    flag=0
    for token in vocab:
        ratio = SequenceMatcher(None, word, token).ratio()
#         sub_ind = word.find(token)
#         if sub_ind == 0 or sub_ind+len(token)==len(word):
#             ratio+=len(token)*0.3/len(word)
        if ratio>max_ratio:
            max_ratio = ratio
            max_word = token
    if max_ratio < 0.8:
        #search in word2vec and glove overlap
#         max_ratio=0.0
#         max_word=''
#         for token in overlap_vocab:
#             ratio = SequenceMatcher(None, word, token).ratio()
# #             sub_ind = word.find(token)
# #             if sub_ind == 0 or sub_ind+len(token)==len(word):
# #                 ratio+=len(token)*0.3/len(word)
#             if ratio>max_ratio:
#                 max_ratio = ratio
#                 max_word = token
#         if max_ratio <0.8:
#             max_word = 'UNK'
#             flag=3
#         else:
#             flag=2
        max_word=word
    else:
        flag=1

    print 'searching for.....', word, ' -> ', max_word, '(', flag,')'
    return max_word



def handle_file(names, vocab, glove_vocab, overlap_vocab):
    rareword2match={}
    for name in names:
        filename= "/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/"+name+'.txt'
        print filename
        readfile=open(filename, 'r')
        writefile=open("/mounts/data/proj/wenpeng/Dataset/StanfordEntailment/"+name+'.norm.to.word2vec.vocab.txt', 'w')
        line_co=0
        for line in readfile:
            parts=line.strip().lower().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
            if len(parts)==3:
                label=parts[0]  # keep label be 0 or 1
                sentence_wordlist_l= re.split('\s+|-',parts[1].strip().lower())#  parts[1].strip().lower().split()
                sentence_wordlist_r= re.split('\s+|-',parts[2].strip().lower())##parts[2].strip().lower().split()
                l_vocab = set(sentence_wordlist_l)
                r_vocab = set(sentence_wordlist_r)
                writefile.write(label+'\t')
                norm_l=[]
                for word in sentence_wordlist_l:
                    if word in vocab or word in glove_vocab:
                        norm_l.append(word)
                    else:
                        if len(word)>0:
                            expected_match = rareword2match.get(word)
                            if expected_match is None:
                                expected_match = find_top_similar_vocab(word, r_vocab, overlap_vocab)

                                rareword2match[word] = expected_match
                            norm_l.append(expected_match)
                        
                norm_r=[]
                for word in sentence_wordlist_r:
                    if word in vocab or word in glove_vocab:
                        norm_r.append(word)
                    else:
                        if len(word)>0:
                            expected_match = rareword2match.get(word)
                            if expected_match is None:
                                expected_match = find_top_similar_vocab(word, l_vocab, overlap_vocab)
                                
                                rareword2match[word] = expected_match
                            norm_r.append(expected_match)
                writefile.write(' '.join(norm_l)+'\t'+' '.join(norm_r)+'\n')
#             line_co+=1
#             if line_co%1==0:
#                 print line_co,' ...'
        writefile.close()
        print 'file norm over'
        readfile.close()



if __name__ == '__main__':
    word2vec=load_word2vec()
    vocab=set(word2vec.keys())
    glove2vec=load_word2vec_file('glove.6B.300d.txt')
    glove_vocab=set(glove2vec.keys())
    overlap_vocab = vocab&glove_vocab
    handle_file(['test', 'train','dev'], vocab,glove_vocab, overlap_vocab)
#     lit='we are your-s   most famous-start . '
#     print re.split('\s+|-',lit)
