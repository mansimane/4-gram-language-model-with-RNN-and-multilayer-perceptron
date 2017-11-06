import re
import pickle
def tokenizeDoc(cur_doc):
    return re.findall('\\w+',cur_doc)

def save_obj(obj, name, epoch =-1):
    if epoch is -1:
        epoch = ''
    with open('obj/'+ name + epoch +'.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,epoch =-1):
    if epoch is -1:
        epoch = ''
    with open('obj/' + name + epoch + '.pkl', 'rb') as f:
        return pickle.load(f)