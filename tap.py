from tdc.utils import retrieve_label_name_list
label_list = retrieve_label_name_list('TAP')

from tdc.single_pred import Develop

data = Develop(name = 'TAP', label_name = label_list[3])

split = data.get_split()

train = split['train']
valid = split['valid']
test = split['test']


import pdb; pdb.set_trace()