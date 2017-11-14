import codecs
import re
import numpy
import operator
import string
import random
import xml.etree.ElementTree as ET
import nltk.data
import nltk
import json

dataset_root = '/save/wenpeng/datasets/NYT/test.json'

#{"em1Text": "New Hampshire", "em2Text": "Cannon Mountain", "label": "/location/location/contains"}

global_rel_map = {'/location/administrative_division/country': 'is an administrative division of',
'/location/neighborhood/neighborhood_of': 'is in the neighborhood of',
'/location/country/capital': "'s capital is",
'/location/location/contains': 'contains',
'/location/country/administrative_divisions': 'has an administrative division called',
#people to company
'/business/person/company': 'works at company',
'/business/company/founders': 'is founded by',
#people to location
'/people/person/nationality': "'s nationality is",
'/people/person/place_of_birth': 'is born in',
'/people/person/place_lived': 'lives in',
'/people/deceased_person/place_of_death': 'died in',
#people vs people
'/people/person/children': 'is the parent of'}

rel_replace_map = {'/location/administrative_division/country': ['is in the neighborhood of',"'s capital is",'contains','has an administrative division called'],
'/location/neighborhood/neighborhood_of': ['is an administrative division of',"'s capital is",'contains','has an administrative division called'],
'/location/country/capital': ['is an administrative division of','is in the neighborhood of'],
'/location/location/contains': ['is an administrative division of','is in the neighborhood of',"'s capital is"],
'/location/country/administrative_divisions': ['is an administrative division of','is in the neighborhood of',"'s capital is"],
#people to company
'/business/person/company': ['is the founder of','does not work at company'],
'/business/company/founders': ['has an employee','is not founded of'],
#people to location
'/people/person/nationality': ['is born in','lives in','died in'],
'/people/person/place_of_birth': ["'s nationality is",'lives in','died in'],
'/people/person/place_lived': ["'s nationality is",'is born in','died in'],
'/people/deceased_person/place_of_death': ["'s nationality is",'is born in','lives in'],
#people vs people
'/people/person/children': ['is the brother of','is the sister of','is the kid of']}

# rel_convert_set = set(['/location/country/capital'])

def statistics(filename):
    data_file = codecs.open(filename, 'r', encoding='utf-8')
    # write_file = codecs.open('/save/wenpeng/datasets/NYT/test_into_sentStatements_twoEntityExist.txt', 'w',encoding='utf-8')
    mention_size = 0
    rel_type_set = set()
    write_size = 0
    label2entSet = {}
    ent2label ={}
    for line in data_file:
        data_dict = json.loads(line)
        sent = data_dict.get('sentText')

        relationMention_list = data_dict.get('relationMentions')
        entMention_list = data_dict.get('entityMentions')
        for entMention in entMention_list:
            ent_text = entMention.get('text')
            ent_label = entMention.get('label')
            ent2label[ent_text] = ent_label
            existing_ents = label2entSet.get(ent_label,[]) # is a list
            existing_ents.append(ent_text)
            label2entSet[ent_label] = existing_ents
    data_file.close()

    data_file = codecs.open(filename, 'r', encoding='utf-8')
    write_file = codecs.open('/save/wenpeng/datasets/NYT/test_into_sentStatements_twoEntityExist.txt', 'w',encoding='utf-8')
    for line in data_file:
        data_dict = json.loads(line)
        sent = data_dict.get('sentText').strip()

        relationMention_list = data_dict.get('relationMentions')
        size_i =  len(relationMention_list)
        for i in range(size_i):
            label = relationMention_list[i].get('label')
            if label !='None':
                # label = label[label.rfind('/')+1:]
                rel_type_set.add(label)
                head_ent = relationMention_list[i].get('em1Text')
                tail_ent = relationMention_list[i].get('em2Text')

                label_description = global_rel_map.get(label)

                pos_statement = head_ent+' '+label_description+' '+ tail_ent
                write_file.write(sent+'\t'+pos_statement+'\t'+str(1)+'\n')
                #create negative statements_list
                #first nega relations
                rel_replaces = rel_replace_map.get(label)
                for rel_rep_description in rel_replaces:
                    nega_rel_statement = head_ent+' '+rel_rep_description+' '+ tail_ent
                    write_file.write(sent+'\t'+nega_rel_statement+'\t'+str(0)+'\n')

                #second, nega entities
                nega_1_statement = tail_ent+' '+label_description+' '+ head_ent
                head_type = ent2label.get(head_ent, 'LOCATION')
                head_replace =   random.sample(set(label2entSet.get(head_type))-set([head_ent]), 1)[0]
                tail_type = ent2label.get(tail_ent, 'LOCATION')

                # print 'tail_ent:', tail_ent
                # print 'tail_type:', tail_type
                # print 'label2entSet.get(tail_type):',label2entSet.get(tail_type)
                tail_replace =   random.sample(set(label2entSet.get(tail_type))-set([tail_ent]), 1)[0]
                nega_2_statement = head_replace+' '+label_description+' '+ tail_ent
                nega_3_statement = head_ent+' '+label_description+' '+ tail_replace
                nega_4_statement = head_replace+' '+label_description+' '+ tail_replace

                write_file.write(sent+'\t'+nega_1_statement+'\t'+str(0)+'\n')
                # write_file.write(sent+'\t'+nega_2_statement+'\t'+str(0)+'\n')
                # write_file.write(sent+'\t'+nega_3_statement+'\t'+str(0)+'\n')
                # write_file.write(sent+'\t'+nega_4_statement+'\t'+str(0)+'\n')



                write_size+=5
    print 'write_size:', write_size
    print 'rel size:', len(rel_type_set)
    for rel in rel_type_set:
        print rel
    write_file.close()
    data_file.close()



if __name__ == '__main__':
    statistics(dataset_root)
