
import os
from datetime import datetime

import numpy as np
import pandas as pd
from pigeon import annotate


def get_annotation(catalog, prms, data):
    
    annots = []
    filepath = os.path.splitext(catalog['annots'])
    myfile, myext = filepath[0], filepath[1]
    if prms['annotation'] == 'load':
        annot_path = os.path.split(catalog['annots'])[0]
        files = os.listdir(annot_path)
        files = [file for file in files if file.startswith('annots')]
        latest = annot_path + '/' + files[-1]
        annots = pd.read_excel(latest).to_records(index=False).tolist()
    else:
        print('WARNING: you must either "load" or "do" annotations')    
    return annots,myfile,myext

def index_annots(prms, sample, annots):
    """Map annotations to initial data observations indices

    Args:
        prms ([type]): [description]
        sample ([type]): [description]
        annots ([type]): [description]

    Returns:
        [type]: [description]
    """
    if prms['annotation'] == 'do':
        indexed_annots = [tuple(np.insert(ann_i, 0, sample['index'][ix])) for ix, ann_i in enumerate(annots)]
    else:
        indexed_annots = annots
    annots_df = pd.DataFrame(indexed_annots, columns = ['index', 'VP', 'annots'])
    return annots_df

def write_annotation(catalog, prms, annots_df, myfile, myext):

    # add current time to filename 
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace(' ','_').replace(':','_').replace('/','_')
    if prms['annotation']=='do' and not os.path.isfile(catalog['annots']):
        annots_df.to_excel(f'{myfile}_{now}{myext}', index=False)
    else:
        print('WARNING: Annots was not written. To write, delete existing and rerun.')

    return annots_df
