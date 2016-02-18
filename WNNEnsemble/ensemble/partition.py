'''
@author: filipi
'''

# Metodos para dividir as instancias e as features


import wann.crossval as w_xval
import wann.encoding as wann_enc
import wann.datafeed as wann_df
import data_process.file_reader as f_rdr

def cria_folds(selected_features, data_name_dict, nmbr_neurons_list, n_folds = 10):
    bse_dict = {}
    tff_dict = {}
    fold_data_dict = {}
    # iterar em todos os conjuntos de features, agora equivale a todos os classificadores
    for z, conj_feat in enumerate(selected_features):# esse z vai simular o numero de neuronios
        
        # criar a chave que identifica unicamente cada classificador (features+neuronios)
        chave = str(z)
        # atencao ao mapear o numero de neuronios
        idx_nmr_neuro = (z % len(nmbr_neurons_list))
        neuronios = nmbr_neurons_list[idx_nmr_neuro]
        bse_dict[chave] = wann_enc.BitStringEncoder(neuronios)
        #usar datadict
        data = f_rdr.read_dataset(data_name_dict[chave]) 
        tff_dict[chave] = wann_df.TxtFileFeed(data, conv=lambda x: (x[0], bse_dict[chave](x[1])))
        fold_data_dict[chave] = w_xval.make_folds(list(tff_dict[chave]), n_folds)
        data.seek(0)
    return bse_dict, tff_dict, fold_data_dict
