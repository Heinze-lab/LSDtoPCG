from segtopcg.utils.utils_supervoxel import get_nbit_chunk_coord, get_chunk_coord, get_chunkId
from segtopcg.utils.utils_local import write_chunk_edges_local

import daisy
import inspect
import json
import logging
import numpy as np
import os
import pandas as pd
import pymongo
import sys
import time

from cloudfiles import CloudFiles
from datetime import date
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds
from pychunkedgraph.io.edges import put_chunk_edges
from pychunkedgraph.graph.edges import Edges
from pychunkedgraph.graph.edges import EDGE_TYPES
from bson.errors import InvalidDocument
from pymongo.errors import OperationFailure, DocumentTooLarge

QUERY_TOO_BIG = (InvalidDocument, OperationFailure, DocumentTooLarge)

logging.basicConfig(level=logging.INFO)

# MongoDB edges to protobuf files:
# one file = one chunk
# Use RAG + roi
# Use put_edges... from PCG

    
def edges_to_graphene_blockwise(fragments_file,
                                cloudpath,
                                db_host,
                                db_name,
                                chunk_voxel_size,
                                edges_collection,
                                edges_dir_local,
                                num_workers,
                                write_local=True,
                                overwrite=False,
                                start_over=False,
                                edges_dir_cloud='edges',
                                upload=True
                               ):
    
    '''
    Start blockwise translation of edges into graphene format, and upload to Google cloud bucket (by default in "edges" dir). 

    Args:
    
        fragments_file (``str``):
        
            Path to the fragments zarr container. By default, the dataset is expected to be named "frags".
    
        cloudpath (``str``):
        
            Path to a Google cloud bucket to upload to. 
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
        
        chunk_voxel_size ([3] list of ``int``):
        
            Size of a chunk in number of voxels (XYZ).
            
        edges_collection (``str``):
        
            MongoDB collection containing edges to translate and upload.
            
        edges_dir_local (``str``):
        
            Local directory where to save edges in protobuf format.
            
        num_workers (``int``):
        
            Number of workers to distribute the tasks to.
            
        write_local (``bool``):
        
            Whether to write edges to local destination.
            
        overwrite (``bool``):
        
            True: edges will be uploaded even if they exist at upload location. 
            False: process will be exited if components exist at upload location.
                        
        start_over (``bool``):
        
            True: progress will be wiped to start from scratch.
            False: will start from where we left off and skip processed edges, based on progress db.

        edges_dir_cloud (``str``):
        
            Name of the edges directory in the cloud bucket. "edges" by default.
    '''

    assert upload or write_local, 'Either upload edges or write them locally, or both'

    if write_local:
        assert os.path.exists(edges_dir_local), 'Edges directory does not exist'
        edges_dir_local = os.path.join(edges_dir_local, db_name)
        os.makedirs(edges_dir_local, exist_ok=True)

    if upload:
        # Check that folder exists at cloudpath. If not, fragments have not been uploaded yet or could be a typo.
        bucket = CloudFiles(cloudpath[:cloudpath.rfind('/')])
        if not bucket.isdir(cloudpath[cloudpath.rfind('/')+1:]):
            print(f'Bucket does not exist at provided location: {cloudpath}')
            print('Aborting.')
            sys.exit()

        # Check if edges already exist at destination
        cf = CloudFiles(cloudpath)
        edges_dir_cloud = '/'.join([cf.cloudpath, edges_dir_cloud])
        if cf.isdir(edges_dir_cloud):
            if overwrite:
                print(f'Edges already exist at path {cloudpath}')
                print('Edges will be overwritten')
            else:
                raise RuntimeError(f'Edges already exist at path {cloudpath}')
    else:
        edges_dir_cloud = None
            
    fragments = open_ds(fragments_file, 'frags')
    chunk_size = Coordinate(chunk_voxel_size) * fragments.voxel_size
    total_roi = fragments.roi
    voxel_size = fragments.voxel_size

    bits_per_chunk_dim = get_nbit_chunk_coord(fragments, chunk_size)    

    logging.info('Found fragments file at:')
    logging.info(f'{fragments_file}')
    logging.info(f'Total ROI: {total_roi}')

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    if 'blocks_edges_in_PCG' not in db.list_collection_names():
        blocks_in_PCG = db['blocks_edges_in_PCG']
        blocks_in_PCG.create_index(
                                   [('block_id', pymongo.ASCENDING)],
                                   name = 'block_id')
    elif start_over:
        db.drop_collection('blocks_edges_in_PCG')
        blocks_in_PCG = db['blocks_edges_in_PCG']
        blocks_in_PCG.create_index(
                                   [('block_id', pymongo.ASCENDING)],
                                   name = 'block_id')
    else:
        blocks_in_PCG = db['blocks_edges_in_PCG']
  
    logging.info('Starting block-wise translation...')
    logging.info(f'Translating using db: {db_name}')
    logging.info(f'Destination: {edges_dir_cloud}')

    task = daisy.Task(
                task_id = f'upload_edges_{db_name}',
                total_roi = fragments.roi,
                read_roi = Roi((0,0,0), chunk_size),
                write_roi = Roi((0,0,0), chunk_size),
                process_function = lambda: translate_edges_worker(
                                                    db_host,
                                                    db_name,
                                                    edges_collection,
                                                    total_roi,
                                                    voxel_size,
                                                    chunk_size,
                                                    bits_per_chunk_dim,
                                                    edges_dir_cloud,
                                                    edges_dir_local,              
                                                    write_local,
                                                    num_workers,
                                                    upload),
                check_function = lambda b: check_block(
                                                    blocks_in_PCG,
                                                    b),
                num_workers = num_workers,
                read_write_conflict = False,
                fit = 'shrink'
                       )
    daisy.run_blockwise([task])
    
    info_ingest = db['info_ingest']

    doc_ingest = {
              'task': 'edges',
              'cloudpath': edges_dir_cloud,
              'frag_file': fragments_file,
              'chunk_voxel_size': list(chunk_voxel_size),
              'date': date.today().strftime('%d%m%Y')
                 }

    info_ingest.insert_one(doc_ingest)
    logging.info(f'Edges were uploaded at {edges_dir_cloud}')
    

def translate_edges_worker(db_host,
                           db_name,
                           edges_collection,
                           total_roi,
                           voxel_size,
                           chunk_size,
                           bits_per_chunk_dim,
                           edges_dir_cloud,
                           edges_dir_local,
                           write_local,
                           num_workers,
                           upload=True,
                           block=None
                          ):

    
    '''
    Worker script translating edges into graphene format, and uploading to Google cloud bucket (by default in "edges" dir). 
    All coordinates are zyx until we upload the data.

    Args:
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
            
        edges_collection (``str``):
        
            MongoDB collection containing edges to translate and upload.
            
        total_roi (`class:Roi`):
        
            Total region of interest covered by the dataset.
        
        voxel_size ([3] list of ``int``):
        
            Resolution in world units (ZYX).
        
        chunk_size ([3] list of ``int``):
        
            Size of a chunk in world units (ZYX).
            
        bits_per_chunk_dim (``int``):
        
            Number of bits used to encode chunk ID
            
        edges_dir_cloud (``str``):
        
            Name of the edges directory in the cloud bucket. 'edges' by default.
            
        edges_dir_local (``str``):
        
            Local directory where to save edges in protobuf format.
            
        write_local (``bool``):
        
            Whether to write edges to local destination.
            
        num_workers (``int``):
        
            Number of workers to distribute the tasks to.
            
    '''

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    trs_coll = db['ids_to_graphene']
    blocks_in_PCG = db['blocks_edges_in_PCG']
    edges_coll = db[edges_collection]
    
    client = daisy.Client()
    
    while True:
            
        with client.acquire_block() as block:

            if block is None: 
                break

            start = time.time()
            roi = block.read_roi  

            chunk_coord = get_chunk_coord(fragments=None,
                                          chunk_roi=roi, 
                                          chunk_size=chunk_size,
                                          total_roi=total_roi)

            # Gather fragment IDs and info for chunks at the interface of main chunk
            chunk_list = [chunk_coord]
            adj_frag_ids = []
            for d in [-1, 1]:
                for dim in range(3):
                    diff = np.zeros([3], dtype=int)
                    diff[dim] = d
                    adj_chunk_coord = chunk_coord + diff

                    if not np.any(adj_chunk_coord < 0):
                        chunk_list.append(adj_chunk_coord.tolist())

            chunk_ids = get_chunkId(bits_per_chunk_dim, 
                                    chunk_coord = chunk_list)

            # Produce a look-up table with initial_ids and their translation to graphene
            # From here we use pandas because it is much faster
            try:
                query = list(trs_coll.find({'graphene_chunk_id':{'$in': chunk_ids.tolist()}},
                                           {'_id':0,
                                            'graphene_chunk_id':1, 
                                            'initial_ids':1, 
                                            'graphene_ids':1}))
            except QUERY_TOO_BIG:
                logging.warning(f'Chunk {chunk_coord}: LUT query too large, splitting per chunk')
                # Query size might exceed mongodb limit, so we split it
                query = []
                for chunk in chunk_ids:
                    query += list(trs_coll.find({'graphene_chunk_id': int(chunk)},
                                                {'_id':0,
                                                 'graphene_chunk_id':1, 
                                                 'initial_ids':1, 
                                                 'graphene_ids':1}))

            nodes_info = []
            for dic in query:
                nodes_info.append(np.vstack([np.repeat(dic['graphene_chunk_id'], len(dic['initial_ids'])), dic['initial_ids'], dic['graphene_ids']]).T)

            translation_table = pd.DataFrame(
                np.vstack(nodes_info), 
                columns=['graphene_chunk_id', 'initial_id', 'graphene_id'],
                dtype=np.uint64
                )
            translation_table.drop(translation_table[translation_table.initial_id == 0].index, inplace=True)

            # Drop duplicates that would be created from having multiple documents for the same chunk
            translation_table = translation_table.drop_duplicates(subset=['graphene_chunk_id', 'initial_id'])

            ##### Query edges #####

            main_chunk_id = chunk_ids[0]
            main_chunk_frag_ids = translation_table.loc[translation_table.graphene_chunk_id == main_chunk_id].initial_id.to_list()

            # Separate queries to fit in memory (limit to query size in mongodb)
            query_in = list(edges_coll.find({'$and': [{'u':{'$in': main_chunk_frag_ids}}, 
                                                      {'v':{'$in': main_chunk_frag_ids}}]},
                                            {'_id': 0, 
                                             'u': 1,
                                             'v': 1,
                                             'merge_score': 1}))

            adj_frag_ids = translation_table.loc[translation_table.graphene_chunk_id != main_chunk_id].initial_id.to_list()
            ### From main chunk to adjacent
            try:
                query_bt_from = list(edges_coll.find({'$and':[{'u':{'$in': main_chunk_frag_ids}}, 
                                                              {'v':{'$in': adj_frag_ids}}]},
                                                     {'_id': 0,
                                                      'u': 1,
                                                      'v': 1,
                                                      'merge_score': 1}))
            except QUERY_TOO_BIG:
                logging.warning(f'Chunk {chunk_coord}: bt_from query too large, splitting per chunk')
                query_bt_from = []
                for chunk in chunk_ids[1:]:
                    query_bt_from += list(edges_coll.find({'$and':[{'u':{'$in': main_chunk_frag_ids}}, 
                                                                   {'v':{'$in': translation_table.loc[translation_table.graphene_chunk_id.eq(chunk)].initial_id.to_list()}}]},
                                                          {'_id': 0,
                                                           'u': 1,
                                                           'v': 1,
                                                           'merge_score': 1}))

            ### To main chunk from adjacent
            try:
                query_bt_to = list(edges_coll.find({'$and':[{'u':{'$in': adj_frag_ids}}, 
                                                            {'v':{'$in': main_chunk_frag_ids}}]},
                                                   {'_id': 0,
                                                    'u': 1,
                                                    'v': 1,
                                                    'merge_score': 1}))
            except QUERY_TOO_BIG:
                logging.warning(f'Chunk {chunk_coord}: bt_to query too large, splitting per chunk')
                query_bt_to = []
                for chunk in chunk_ids[1:]:
                    query_bt_to += list(edges_coll.find({'$and':[{'u':{'$in': translation_table.loc[translation_table.graphene_chunk_id.eq(chunk)].initial_id.to_list()}}, 
                                                                 {'v':{'$in': main_chunk_frag_ids}}]},
                                                        {'_id': 0,
                                                         'u': 1,
                                                         'v': 1,
                                                         'merge_score': 1}))
                
            # Edges in chunk
            # in_chunk: edges between supervoxels within a chunk
            in_edges_df = pd.DataFrame(query_in, columns=['u', 'v', 'merge_score'])
            in_edges_df['u'] = in_edges_df['u'].astype(np.uint64)
            in_edges_df['v'] = in_edges_df['v'].astype(np.uint64)
            in_edges_df['merge_score'] = in_edges_df['merge_score'].astype(float)

            in_edges_df = in_edges_df.merge(translation_table.loc[translation_table.graphene_chunk_id == main_chunk_id], 'left', left_on='u', right_on='initial_id') \
                                .drop(columns='initial_id') \
                                .rename(columns={'graphene_chunk_id': 'u_graphene_chunk_id', 'graphene_id': 'u_graphene_id'})
            in_edges_df = in_edges_df.merge(translation_table.loc[translation_table.graphene_chunk_id == main_chunk_id], 'left', left_on='v', right_on='initial_id') \
                                .drop(columns='initial_id') \
                                .rename(columns={'graphene_chunk_id': 'v_graphene_chunk_id', 'graphene_id': 'v_graphene_id'})

            in_chunk_edges = Edges(*in_edges_df[['u_graphene_id', 'v_graphene_id']].to_numpy(dtype=np.uint64).T)
            in_chunk_edges.affinities = (1.0 - in_edges_df['merge_score']).to_numpy(np.float32)

            # If there is a duplicated initial ID, it means it exists in two chunks (split fragment).
            # Each piece is translated to a different graphene ID, linked later by cross edges.
            # For split_fragments, we want to keep all duplicates to later build the cross edges
            # linking the graphene IDs that were created from these duplicates
            split_fragments = translation_table.loc[translation_table.initial_id.duplicated(keep=False)].copy()

            # Edges between chunks
            # between_chunks: edges between supervoxels across chunks
            main_lut = translation_table.loc[translation_table.graphene_chunk_id == main_chunk_id]
            adj_lut = translation_table.loc[translation_table.graphene_chunk_id != main_chunk_id]

            # Orient every edge as (u: main chunk side, v: adjacent chunk side)
            bt_from_df = pd.DataFrame(query_bt_from, columns=['u', 'v', 'merge_score'])
            bt_to_df = pd.DataFrame(query_bt_to, columns=['u', 'v', 'merge_score']).rename(columns={'u': 'v', 'v': 'u'})
            bt_edges_df = pd.concat([bt_from_df, bt_to_df], ignore_index=True)
            bt_edges_df['u'] = bt_edges_df['u'].astype(np.uint64)
            bt_edges_df['v'] = bt_edges_df['v'].astype(np.uint64)
            bt_edges_df['merge_score'] = bt_edges_df['merge_score'].astype(float)

            # u -> its graphene ID in the main chunk (unique per chunk)
            bt_edges_df = bt_edges_df.merge(main_lut[['initial_id', 'graphene_id']]
                                                .rename(columns={'initial_id': 'u', 'graphene_id': 'u_graphene_id'}),
                                            'inner', on='u')
            # v -> its graphene ID in each adjacent chunk it lives in
            # If v is split across several adjacent chunks, the edge is fanned out to every piece
            bt_edges_df = bt_edges_df.merge(adj_lut.rename(columns={'initial_id': 'v',
                                                                    'graphene_id': 'v_graphene_id',
                                                                    'graphene_chunk_id': 'v_chunk'}),
                                            'inner', on='v')

            # Drop edges already represented as in_chunk edges in some chunk:
            # - v also lives in the main chunk -> in_chunk edge of the main chunk
            # - u also lives in v's chunk      -> in_chunk edge of that adjacent chunk
            v_in_main = bt_edges_df.v.isin(main_lut.initial_id).to_numpy()
            u_in_v_chunk = pd.MultiIndex.from_frame(bt_edges_df[['u', 'v_chunk']]) \
                                .isin(pd.MultiIndex.from_frame(adj_lut[['initial_id', 'graphene_chunk_id']]))
            bt_edges_df = bt_edges_df[~(v_in_main | u_in_v_chunk)] \
                                .drop_duplicates(subset=['u_graphene_id', 'v_graphene_id'])

            between_chunk_edges = Edges(*bt_edges_df[['u_graphene_id', 'v_graphene_id']].to_numpy(dtype=np.uint64).T)
            between_chunk_edges.affinities = (1.0 - bt_edges_df['merge_score']).to_numpy(np.float32)

            # Edges cross chunk
            # cross_chunk: edges between parts of the same supervoxel before chunking, split across chunks
            cross_edges_df = split_fragments.loc[split_fragments.graphene_chunk_id == main_chunk_id] \
                                    .merge(split_fragments.loc[split_fragments.graphene_chunk_id != main_chunk_id], 
                                           'inner', left_on='initial_id', right_on='initial_id') \
                                    .rename(columns={'graphene_id_x': 'u_graphene_id', 'graphene_id_y': 'v_graphene_id'})

            cross_chunk_edges = Edges(*cross_edges_df[['u_graphene_id', 'v_graphene_id']].to_numpy(dtype=np.uint64).T)
            cross_chunk_edges.affinities = np.repeat(np.float32('inf'), len(cross_edges_df))

            # Protobuf format
            edges_proto_d = {
                    EDGE_TYPES.in_chunk: in_chunk_edges,
                    EDGE_TYPES.between_chunk: between_chunk_edges,
                    EDGE_TYPES.cross_chunk: cross_chunk_edges
                            }

            if upload:
                put_chunk_edges(edges_dir_cloud, 
                                chunk_coord[::-1], # x,y,z
                                edges_proto_d, 
                                compression_level=22)

            if write_local:      
                logging.info('WRITE')
                write_chunk_edges_local(edges_dir_local, 
                                        chunk_coord[::-1], # x,y,z
                                        edges_proto_d, 
                                        compression_level=22)

            # Write document to keep track of blocks done
            document = {
                   'num_cpus': num_workers,
                   'block_id': block.block_id,
                   'graphene_chunk_coord': list(map(int, chunk_coord[::-1])),
                   'read_roi': (block.read_roi.get_begin(),
                                block.read_roi.get_shape()),
                   'write_roi': (block.write_roi.get_begin(),
                                 block.write_roi.get_shape()),
                   'start': start,
                   'duration': time.time() - start
                        }

            blocks_in_PCG.insert_one(document)


def check_block(blocks_in_PCG, block):

    done = blocks_in_PCG.count_documents({'block_id': block.block_id}) >=1

    return done


if __name__ == '__main__':
    
    config_file = sys.argv[1]
    
    with open(config_file, 'r') as f:
        config = json.load(f)

    params = inspect.signature(edges_to_graphene_blockwise).parameters
    relevant_args = {k: v for k, v in config.items() if k in params}
    
    edges_to_graphene_blockwise(**relevant_args)