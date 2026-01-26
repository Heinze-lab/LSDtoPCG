from segtopcg.utils.utils_supervoxel import get_chunkId, get_chunk_list
from segtopcg.utils.utils_components import *

import traceback
import inspect
import json
import logging
import numpy as np
import networkx as nx
import os
import pymongo
import sys
import time

from cloudfiles import CloudFiles
from collections import defaultdict
from datetime import date
from funlib.geometry import Coordinate
from funlib.persistence import open_ds
from multiprocessing import Pool
from pychunkedgraph.io.components import put_chunk_components

logging.basicConfig(level=logging.INFO)


def upload_components_chunkwise(
                                fragments_file,
                                edges_dir_local,
                                cloudpath,
                                db_host,
                                db_name,
                                chunk_voxel_size,
                                edges_aff_threshold,
                                num_workers,
                                isolate_chunks_mode,
                                group_size,
                                overwrite=False,                    
                                start_over=False,
                                edges_dir_cloud='edges',
                                components_dir_cloud='components',
                                keep_datastack_progress=False
                               ):
    '''
    Start workers in parallel using multiprocessing. Workers compute and upload components per chunk by thresholding edges based on their affinity score. Components can be isolated from each other, either all or based on a metric (not yet implemented).

    Args:
    
        fragments_file (``str``):
        
            Path to the fragments zarr container. By default, the dataset is expected to be named "frags".

        edges_dir_local (``str``):

            Path to the local directory containing edges in protobuf format for all databases.
    
        cloudpath (``str``):
        
            Path to a Google cloud bucket to upload to. 
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset, or None.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
        
        chunk_voxel_size ([3] list of ``int``):
        
            Size of a chunk in number of voxels (ZYX).
            
        edges_aff_threshold (``float``):
        
            Affinity threshold used to compute connected components. Threshold corresponds to an affinity score from 0 (bad) to 1 (good): corresponds to 1-merge_score. Only affinity scores equal or above threshold are kept.
        
        num_workers (``int``):
        
            Number of workers to distribute the tasks to.
        
        isolate_chunks_mode (``str``):
        
            Mode to use to isolate chunks. If equivalent to False, chunks will not be isolated. See isolate_chunks function for modes description.
            
        group_size (``int``):
            
            Number of chunks to group together in each dimension when chunks are being isolated.
            
        overwrite (``bool``):
        
            True: components will be uploaded even if they exist at upload location. 
            False: process will be exited if components exist at upload location.
            
        start_over (``bool``):
        
            True: progress will be wiped to start from scratch.
            False: will start from where we left off and skip uploaded components, based on progress db.
            
        edges_dir_cloud (``str``):
        
            Name of the edges directory in the cloud bucket. "edges" by default.
            
        components_dir_cloud (``str``):
        
            Name of the components directory in the cloud bucket. "components" by default.
    '''

    # Variables
    bucket = CloudFiles(cloudpath[:cloudpath.rfind('/')])
    fragments = open_ds(fragments_file, 'frags')
    chunk_size = fragments.voxel_size*Coordinate(chunk_voxel_size)
    cf = CloudFiles(cloudpath)
    edges_dir_cloud =  '/'.join([cloudpath, edges_dir_cloud])
    components_dir = '/'.join([cloudpath, components_dir_cloud])
    edges_dir_local = os.path.join(edges_dir_local, db_name)
    
    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    # Assertions
    if not bucket.isdir(cloudpath[cloudpath.rfind('/')+1:]):
        raise RuntimeError(f'Bucket does not exist at provided location: {cloudpath}')
    if not cf.isdir(edges_dir_cloud.split('/')[-1]) or not len(os.listdir(edges_dir_local)) > 0: 
        raise RuntimeError('Edges were not computed and/or saved')
    if cf.isdir(components_dir.split('/')[-1]) and not overwrite: 
        raise RuntimeError(f'Components already exist at {components_dir}')
    
    if start_over:
        logging.info('Deleting info database and starting over...')
        db['components_info'].drop()
    
    logging.info(f'Components will be uploaded at {components_dir}')
    if keep_datastack_progress:
        logging.info('Will keep saved progress')
    
    # Prepare list of chunks to isolate
    chunk_list, n_chunks = get_chunk_list(fragments_file, chunk_size)
    
    if not isolate_chunks_mode:
        chunks_to_cut = []
    elif isinstance(isolate_chunks_mode, str):
        chunks_to_cut = isolate_chunks(isolate_chunks_mode,
                                       chunk_list,
                                       db_host,
                                       db_name)
    else:
        raise RuntimeError('Specify isolate_chunks_mode')

    # Prepare inputs for multiprocessing
    inputs = [[chunk_coord,
               components_dir,
               edges_dir_local,
               edges_aff_threshold,
               db_host,
               db_name,
               chunks_to_cut,
               group_size,
               keep_datastack_progress
              ] for chunk_coord in chunk_list]

    try:
        logging.info(f'Uploading components for {np.product(n_chunks)} chunks...')
        
        with Pool(num_workers) as pool:
            results = pool.starmap(connected_components_to_cloud_worker, inputs)
        logging.info(f'{np.sum(results)} components were uploaded at {components_dir}')
        logging.info(f'{np.sum(np.logical_not(results))} components were skipped')

    except Exception as e:
        print('Uploading components failed')
        print(f'Error: {e}')
        traceback.print_exc()
        sys.exit()
           
    info_ingest = db['info_ingest']
    doc_ingest = {
                  'task': 'components',
                  'cloudpath': components_dir,
                  'frag_file': fragments_file,
                  'chunk_voxel_size': list(chunk_voxel_size),
                  'edges_threshold': float(edges_aff_threshold),
                  'isolate_chunks_mode': str(isolate_chunks_mode),
                  'group_size': group_size,
                  'date': date.today().strftime('%d%m%Y')
                 }
    info_ingest.insert_one(doc_ingest)


def connected_components_to_cloud_worker(chunk_coord,
                                         components_dir,
                                         edges_dir,
                                         edges_threshold,
                                         db_host,
                                         db_name,
                                         chunks_to_cut,
                                         group_size,
                                         keep_datastack_progress
                                        ):
    '''
    Worker script. Computes connected components by filtering edges based on an affinity threshold.
    
    Args:
    
        chunk_coord ([3] list of ``int``):
        
            Coordinates of the chunk (XYZ) to process.
                
        components_dir (``str``):
        
            Path to the location where to write components in the Google cloud bucket. 
            
        edges_dir (``str``):
        
            Local path to the directory containing edges in protobuf format. 
            
        edges_threshold (``float``):
        
            Affinity threshold used to compute connected components. Threshold corresponds to an affinity score from 0 (bad) to 1 (good): corresponds to 1-merge_score. Only affinity scores equal or above threshold are kept.
            
        db_host (``str``):
        
            URI to the MongoDB containing information relevant to the isolation mode.
            
        db_name (``str``):
        
            MongoDB database containing information relevant to the isolation mode.
            
        chunks_to_cut ([n,3] list of ``int``):
        
            List of chunk coordinates (XYZ) to be isolated from neighbors. Chunk process will be isolated if contained in this list, or will be isolated from a neighbor if neighbor is contained in this list.

        group_size (``int``):
            
            Number of chunks to group together in each dimension when chunks are being isolated.
    '''
    
    # Variables
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    info = db['components_info']
    bits_per_chunk_dim = db['info_ingest'].find_one({'task': 'supervoxels'}, {'spatial_bits':1})['spatial_bits']
    main_chunk = np.array(chunk_coord, dtype=int) # xyz
    
    # Check if chunk was processed
    if check_block(info, main_chunk):
        print(f'chunk_coord: {chunk_coord} skipped')
        return False
        
    # Obtain adjacent chunks to include
    adj_chunk_list = get_adjacent_chunk_coords(main_chunk, chunks_to_cut, group_size)
    
    # Obtain all edges crossing into main chunk from any adjacent chunk considered
    edges, scores = get_main_chunk_edges(edges_dir, main_chunk, adj_chunk_list, bits_per_chunk_dim)

    components = []
    if keep_datastack_progress:
        # If this is a re-ingest of a dataset that was worked on previously
        # Requires to have ran download_components_progress.py
        main_chunk_id = get_chunkId(bits_per_chunk_dim, chunk_coord=main_chunk[::-1])
        progress = db['info_progress_save'].find_one({'chunk_id': int(main_chunk_id)})
        if progress is not None:
            components = progress['components']

            # Remove these components from edges so that they don't mixed up
            dump_nodes = np.concatenate(components).astype(np.uint64)
            mask = ~np.isin(edges, dump_nodes).any(1)
            edges = edges[mask]
            scores = scores[mask]
    
    if edges.size:
        # Threshold edges
        # Only keep if affinity score >= threshold
        thresh_edges = edges[scores >= edges_threshold]

        # Get connected components
        G = nx.Graph()
        G.add_edges_from(thresh_edges)
        
        # Nodes include even thresholded edges because they exist but are isolated
        nodes = np.unique(edges)

        G.add_nodes_from(nodes) # Include isolated nodes
        components += [list(map(int, c)) for c in list(nx.connected_components(G))]
   
    # Upload
    if components:
        put_chunk_components(components_dir, 
                             components,  
                             chunk_coord)

        print(f'chunk_coord: {chunk_coord} uploaded')
    else:
        print(f'No components at {chunk_coord}')
        
    document = {
                'graphene_chunk_coord': main_chunk.tolist(), # xyz
                'threshold': edges_threshold,
                'no_components': not components,
                'time': time.time()
                }
    info.insert_one(document)

    return True
    

if __name__ == '__main__':
    
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    params = inspect.signature(upload_components_chunkwise).parameters
    relevant_args = {k: v for k, v in config.items() if k in params}

    upload_components_chunkwise(**relevant_args)