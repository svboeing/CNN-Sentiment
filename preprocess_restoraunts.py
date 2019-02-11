import numpy as np
import re
import xml.etree.ElementTree as ET
tree = ET.parse('ABSA-15_Restaurants_Train_Final.xml')
root = tree.getroot()

sents = []
opins = []
for review in root:
    for sentence in review[0]:
        #sentence[0] #text
        if len(sentence) > 1:
            _ = re.sub(r"[\.\-\?\!,\"\']", " ", sentence[0].text.lower()).strip().split(' ')
            __ = []
            for word in _:
                if word != "":#get rid of ""s
                    __.append(word)
                    #if word in word_id:
                    #    __.append(word_id[word])
                    #else:
                    #    __.append(0)

            if len(__) != 0:
                sents.append(__)
                opinions = [opin.attrib for opin in sentence[1]]
                opins.append(opinions)
            #data[sentence[0].text] = [opin.attrib for opin in sentence[1]]
_ = sorted(zip(sents,opins), key=lambda pair: len(pair[0]))
sents, opins = zip(*_)
classes_labels = {'positive': 1, 'negative': -1, 'neutral': 0}
dumb_Y = []
for opin in opins:
    _ = [target_op['polarity'] for target_op in opin]
    __ = np.asarray([classes_labels[class_] for class_ in _], dtype='int32')
    dumb_Y.append(round(np.mean(__, dtype='int32')))  # labels are crushed into a mean


#print(opins[120:125], dumb_Y[120:125])

print(sents[140:147], dumb_Y[140:147])