
from tdc.single_pred import Develop
from tdc.utils import load as data_load
from tdc.utils import retrieve_label_name_list
import copy

def load_and_process_data(path='/public/home/gongzhichen/code/data', name = 'tap', label_ind = 0):
    if name == 'tap':
        label_list = retrieve_label_name_list(name)
        print('labels for multi-target: ', label_list)

        df = data_load.pd_load(name, path)
        df = df.dropna() 
        df2 = df.loc[:, ~df.columns.duplicated()]  ### remove the duplicate columns
        df = df2
        h_chain = []
        l_chain = []
        for ind in range(len(df)):
            v = df.loc[ind,'X']
            v = v.strip('[\']').split('\\n')
            v[0] = v[0].strip("'")
            v[1] = v[1].strip(" '")
            h_chain.append(copy.deepcopy(v[0])) 
            l_chain.append(copy.deepcopy(v[1]))
            
            v = v[0] + v[1]

            df.loc[ind,'X'] = copy.deepcopy(v)
            
        
        sequences = df['X'].tolist()
        labels = df[label_list[label_ind]].tolist()
        labels = [float(v) for v in labels]
        assert len(sequences) == len(labels)
        import pdb; pdb.set_trace()
        return (sequences, h_chain, l_chain), labels
    elif name == 'sabdab_chen':
        
        df = data_load.pd_load(name, path)
        df = df.dropna() 
        df2 = df.loc[:, ~df.columns.duplicated()]  ### remove the duplicate columns
        df = df2
        h_chain = []
        l_chain = []
        for ind in range(len(df)):
            v = df.loc[ind,'X']
            v = v.strip('[\']').split(',')
            v[0] = v[0].strip("'")
            v[1] = v[1].strip(" '")
            h_chain.append(copy.deepcopy(v[0])) 
            l_chain.append(copy.deepcopy(v[1]))
            
            v = v[0] + v[1]

            df.loc[ind,'X'] = copy.deepcopy(v)
            
        sequences = df['X'].tolist()
        labels = df['Y'].tolist()
        
        assert len(sequences) == len(labels)
        return (sequences, h_chain, l_chain), labels

if __name__ == '__main__':    
    sequences, labels = load_and_process_data(name='tap')
    sequences, h_chain, l_chain = sequences
    import pdb; pdb.set_trace()
    for seq, h_cha, l_cha, label in zip(sequences, h_chain, l_chain, labels):
        assert(seq == h_cha + l_cha)
        
        