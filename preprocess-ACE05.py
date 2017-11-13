
import codecs
import re
import numpy
import operator
import string
import random
import xml.etree.ElementTree as ET
import nltk.data
import nltk
nltk.download()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

dataset_root = '/shared/experiments/hpeng7/data/ACE2005/data/English/'

def parse_a_xml(xmlfile):
    e = ET.parse(xmlfile).getroot()
    entityID2Str={}
    for entity in e.iter('entity_mention'):
        id =  entity.get('ID')
        strr =  entity[1][0].text
        entityID2Str[id]=strr
    # print 'entity load over'

    statements_list = []
    rel_extend_list =[]

    for relation in e.iter('relation'):
        rel_type = relation.get('SUBTYPE')
        for relationMention in relation.findall('relation_mention'):
            args = relationMention.findall('relation_mention_argument')
            relationMention_extend = relationMention[0][0].text
            if len(args)>=2:
                arg_id1 = args[0].get('REFID')
                arg_str1 = entityID2Str.get(arg_id1)
                arg_id2 = args[1].get('REFID')
                arg_str2 = entityID2Str.get(arg_id2)
                # print 'statement:', arg_str1, rel_type, arg_str2
                if arg_str1 is not None and arg_str2 is not None:
                    statements_list.append(arg_str1+' '+ rel_type+' '+ arg_str2)
                    rel_extend_list.append(relationMention_extend)
    return statements_list, rel_extend_list

def parse_a_passage(filename):
    readfile = open(filename, 'r')
    start_flag = False
    passage_str = ''
    for line in readfile:
        if line.strip() == '<TEXT>':
            start_flag=True
        if start_flag:
            if line.find('<')>=0 or line.find('>')>=0:
                continue
            else:
                passage_str+=' '+line.strip()
    readfile.close()
    return passage_str


def iter_filenames():
    folders = ['bc','bn','cts','nw','un','wl']
    write_pairs = open('/home/wyin3/Datasets/ACE05-DocStatements/ACE05-DocStatements.txt', 'w')
    for folder in folders:
        path = dataset_root+folder+'/'
        Filelist = path+'FileList'
        filelist_open = open(Filelist, 'r')
        filenames = []
        for line in filelist_open:
            if line.strip().find('DOCID')>=0 or len(line.strip())==0 or line.strip().find('Total')>=0:
                continue
            # print 'line:', line
            parts=line.strip().split()
            annotator = parts[1].split(',')[0]
            if annotator =='adj':
                filename_statement = path+annotator+'/'+parts[0]+'.apf.xml'
                filename_passage = path+annotator+'/'+parts[0]+'.sgm'
                print 'parsing ... ', path+annotator+'/'+parts[0]
                passage = parse_a_passage(filename_passage)
                statements_list, extention_list = parse_a_xml(filename_statement)
                if len(statements_list)!= len(extention_list):
                    print 'len(statements_list)!= len(extention_list):', len(statements_list),  len(extention_list)
                    exit(0)

                #split passage into a list of sents, then label positive and negative
                sents = tokenizer.tokenize(passage)
                for idd, statement in enumerate(statements_list):
                    for sent in sents:
                        if sent.find(extention_list[idd])>=0:
                            write_pairs.write(sent+'\t'+statement+'\t'+str(1)+'\t'+extention_list[idd]+'\n') #positive
                        else:
                            write_pairs.write(sent+'\t'+statement+'\t'+str(0)+'\t'+extention_list[idd]+'\n') #negative

                print '\t\t\t\t write over'
        filelist_open.close()
    write_pairs.close()

def tokenize(filename):

    readfile = open(filename, 'r')
    writefile = open('/home/wyin3/Datasets/ACE05-DocStatements/ACE05-DocStatements-tokenized.txt', 'w')
    pos_co=0
    total_co=0
    for line in readfile:
        # print 'line:', line
        parts =  line.strip().split('\t')
        if len(parts)==4:

            sent = parts[0]
            statement = parts[1].split()
            label = parts[2]
            if label == '1':
                pos_co+=1
            total_co+=1
            sent_words = nltk.word_tokenize(sent)
            statement_words = [statement[0]]+statement[1].split('-')+[statement[-1]]
            writefile.write(' '.join(sent_words)+'\t'+' '.join(statement_words)+'\t'+label+'\n')

    print 'pos size: ', pos_co, ' ratio:', pos_co*100.0/total_co, '%'
    writefile.close()
    readfile.close()



if __name__ == '__main__':
    # parse_a_xml(dataset_root+'bc/adj/CNN_LE_20030504.1200.02-2.apf.xml')
    # parse_a_passage(dataset_root+'bc/adj/CNN_LE_20030504.1200.02-2.sgm')
    # iter_filenames()
    tokenize('/home/wyin3/Datasets/ACE05-DocStatements/ACE05-DocStatements.txt')
