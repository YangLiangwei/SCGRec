import numpy as np
import pdb
import torch

class NegativeSampler(object):
    def __init__(self, dic):
        self.dic_user_game = dic

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src_type, edge, dst_type = etype
            src, _ = g.find_edges(eids, etype = etype)
            dst = []
            for i in range(src.shape[0]):
                s = int(src[i])
                while True:
                    negitem = np.random.randint(0, g.num_nodes(dst_type))
                    if negitem in self.dic_user_game[s]:
                        continue
                    else:
                        break
                dst.append(negitem)
        dst = torch.tensor(dst)
        result_dict[etype] = (src, dst)
        return result_dict


