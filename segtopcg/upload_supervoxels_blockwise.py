from segtopcg.utils.utils_supervoxel import *

import daisy
import inspect
import json
import logging
import numpy as np
import pymongo
import sys
import time
import traceback

from cloudvolume import CloudVolume
from cloudfiles import CloudFiles
from datetime import date
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds
from funlib.segment.arrays.replace_values import replace_values

logging.basicConfig(level=logging.INFO)

# Hard coded because common to all datasets
# Layer ID = 1 is first layer containing supervoxel IDs
# Other layers are computed during ingest
layer_id = 1
layer_bits = 8

def supervoxel_to_graphene_blockwise(fragments_file, 
                                     cloudpath,
                                     db_host, 
                                     db_name, 
                                     chunk_voxel_size,
                                     cloudvolume_provenance,
                                     num_workers,
                                     overwrite=False,
                                     start_over=False,
                                     supervoxels_dir='fragments',
                                     upload=True
                                    ):

    '''
    Start blockwise translation of supervoxel IDs into graphene format, and upload to Google cloud bucket (by default in "fragments" dir).

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
        
            Size of a chunk in number of voxels (ZYX).
            
        cloudvolume_provenance (``str``):
            
            Description of the volume.
        
        num_workers (``int``):
        
            Number of workers to distribute the tasks to.
            
        overwrite (``bool``):
            
            True: Will upload even if a volume is present at destination.
            False: Will interrupt process if a volume is present a destination.
            
        start_over (``bool``):
        
            True: progress will be wiped to start from scratch.
            False: will start from where we left off and skip processed edges, based on progress db.

        supervoxels_dir (``str``):
        
            Name of the supervoxels directory in the cloud bucket. "fragments" by default.
    '''
    
    # Variables
    fragments = open_ds(fragments_file, 'frags')
    chunk_size = Coordinate(chunk_voxel_size) * fragments.voxel_size
    bits_per_chunk_dim = get_nbit_chunk_coord(fragments, chunk_size)
    
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    
    if 'blocks_translated' not in db.list_collection_names():
        blocks_translated = db['blocks_translated']
        blocks_translated.create_index(
                  [('block_id', pymongo.ASCENDING)],
                  name='block_id')
    elif start_over:
        db.drop_collection('blocks_translated')
        blocks_translated = db['blocks_translated']
        blocks_translated.create_index(
                                       [('block_id', pymongo.ASCENDING)],
                                       name = 'block_id')
    else:
        blocks_translated = db['blocks_translated']

    if 'ids_to_graphene' not in db.list_collection_names():
        ids_to_graphene = db['ids_to_graphene']
        ids_to_graphene.create_index([('graphene_chunk_id', pymongo.ASCENDING)], unique=True)
    elif start_over:
        db.drop_collection('ids_to_graphene')
        ids_to_graphene = db['ids_to_graphene']
        ids_to_graphene.create_index([('graphene_chunk_id', pymongo.ASCENDING)], unique=True)

    # Initiate CloudVolume object
    if upload:
        # Check if volume already exists at destination
        cf = CloudFiles(cloudpath)
        supervoxels_dir = '/'.join([cf.cloudpath, supervoxels_dir])
        if cf.isdir(supervoxels_dir):
            if overwrite:
                print(f'Fragments already exist at path {cloudpath}')
                print('Overlapping fragments volume will be overwritten')
            else:
                raise RuntimeError(f'Fragments already exist at path {cloudpath}')
            
        try:
            logging.info('Volume will be uploaded to the cloud')
            voxel_offset = (fragments.roi.get_begin()//fragments.voxel_size)[::-1]
            logging.info(f'Initializing CloudVolume with voxel offset: {voxel_offset}')
            destination_chunk_size = np.array(chunk_size)[::-1]/2/np.array(fragments.voxel_size)[::-1] # xyz

            info = CloudVolume.create_new_info(
                                num_channels = 1,
                                layer_type = 'segmentation',
                                data_type = 'uint64',
                                encoding = 'compressed_segmentation',
                                resolution = fragments.voxel_size[::-1], # x,y,z voxel_size
                                voxel_offset = voxel_offset, # x,y,z offset in voxel
                                mesh = 'mesh',
                                chunk_size = destination_chunk_size,
                                volume_size = fragments.shape[::-1]) # x,y,z shape in voxel

            vol = CloudVolume(supervoxels_dir, 
                            info = info, 
                            fill_missing = False,
                            parallel = False,
                            progress = False)

            vol.provenance.description = cloudvolume_provenance
            vol.commit_info()
            vol.commit_provenance()
            logging.info(f'CloudVolume inialized at path: {supervoxels_dir}')

        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()
            sys.exit()

        logging.info(f'Volume bounds: {vol.bounds}') 
    else:
        vol = None

    logging.info(f'Starting nodes translation with chunk size: {chunk_size}') 
    
    # Start blockwise translation and upload
    task = daisy.Task(task_id = f'upload_supervoxels_{db_name}',
                        total_roi = fragments.roi,
                        read_roi = Roi((0,0,0), chunk_size),
                        write_roi = Roi((0,0,0), chunk_size),
                        process_function = lambda: upload_supervoxels_worker(
                                                                        fragments_file,
                                                                        bits_per_chunk_dim,
                                                                        chunk_size,
                                                                        vol,
                                                                        db_host,
                                                                        db_name,
                                                                        num_workers,
                                                                        upload),
                        check_function = lambda b: check_block(
                                                        blocks_translated,
                                                        b),
                        num_workers = num_workers,
                        read_write_conflict = False,
                        fit = 'shrink'
                       )
    daisy.run_blockwise([task])

    info_ingest = db['info_ingest']
    doc_ingest = {
                  'task': 'supervoxels',
                  'cloudpath': supervoxels_dir,
                  'frag_file': fragments_file,
                  'chunk_voxel_size': list(chunk_voxel_size),
                  'spatial_bits': bits_per_chunk_dim, 
                  'layer_id_bits': 8,
                  'date': date.today().strftime('%d%m%Y')
                 }
    info_ingest.insert_one(doc_ingest)
    logging.info('Uploading supervoxels complete!')


def upload_supervoxels_worker(fragments_file,
                              bits_per_chunk_dim,
                              chunk_size,
                              vol,
                              db_host,
                              db_name,
                              num_workers,
                              upload=True
                             ):

    '''
    Start blockwise translation of supervoxel IDs into graphene format, and upload to Google cloud bucket (by default in "fragments" dir).
    A look-up table for IDs is stored in MongoDB for future translation of edges.

    Args:
    
        fragments_file (``str``):
        
            Path to the fragments zarr container. By default, the dataset is expected to be named "frags".
    
        bits_per_chunk_dim (``int``):
        
            Number of bits used to encode chunk ID. 
            
        vol (``str``):
        
            CloudVolume object to write to.
            
        db_host (``str``):
        
            URI to the MongoDB instance containing information about the dataset.
            
        db_name (``str``):
        
            MongoDB database containing information about the dataset.
        
        chunk_size ([3] list of ``int``):
        
            Size of a chunk in world units.
        
        num_workers (``int``):
        
            Number of workers to distribute the tasks to. Used by the worker for documenting the task.
    '''
    
    fragments = open_ds(fragments_file, 'frags')
    
    # Initiate MongoDB client and collections
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    blocks_translated = db['blocks_translated']
    ids_to_graphene_collection = db['ids_to_graphene']

    client = daisy.Client()
    
    while True:
    
        with client.acquire_block() as block:

            if block is None:
                break
            
            start = time.time()
            
            data = fragments.intersect(block.read_roi)
            data = data.to_ndarray()
            
            frag_ids = np.unique(data)
            
            # Compute new supervoxel IDs
            chunk_coord = get_chunk_coord(fragments, block.read_roi, chunk_size) # z,y,x
            chunk_id = get_chunkId(bits_per_chunk_dim, 
                                   block.read_roi, 
                                   chunk_size, 
                                   chunk_coord = chunk_coord) # zyx chunk coord
            frag_ids, seg_ids = get_segId(frag_ids)
            
            new_ids = []
            for x in seg_ids:
                if x == 0:
                    nid = np.uint64(0)
                else:
                    nid = chunk_id | np.uint64(x)
                new_ids.append(nid)
            new_ids = np.array(new_ids, dtype=np.uint64)
            
            graphene_data = replace_values(data, frag_ids, new_ids, inplace = False)
                
            # Upload data
            if upload:
                try:
                    z1,y1,x1 = block.read_roi.get_begin()//fragments.voxel_size 
                    z2,y2,x2 = block.read_roi.get_end()//fragments.voxel_size
                    vol[x1:x2, y1:y2, z1:z2] = np.transpose(graphene_data, (2,1,0)) # zyx to xyz
                except Exception as e:
                    print(f'Error: {e}')
                    traceback.print_exc()
    
            document = {
                    'num_cpus': num_workers,
                    'block_id': block.block_id,
                    'graphene_chunk_id': int(chunk_id),    
                    'read_roi': (block.read_roi.get_begin(),
                                 block.read_roi.get_shape()
                                ),
                    'write_roi': (block.write_roi.get_begin(),
                                  block.write_roi.get_shape()
                                 ),
                    'start': start,
                    'duration': time.time() - start
                       }
            
            doc_ids_to_graphene = {
                            'block_id': block.block_id,
                            'graphene_chunk_coord': list(map(int, chunk_coord[::-1])),
                            'graphene_chunk_id': int(chunk_id),
                            'initial_ids': list(map(int, frag_ids)),
                            'graphene_ids': list(map(int, new_ids))
                                  }
            
            blocks_translated.insert_one(document)
            ids_to_graphene_collection.insert_one(doc_ids_to_graphene)


def check_block(blocks_translated, block):

    done = blocks_translated.count_documents({'block_id': block.block_id}) >=1

    return done


if __name__ == '__main__':

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    params = inspect.signature(supervoxel_to_graphene_blockwise).parameters
    relevant_args = {k: v for k, v in config.items() if k in params}
    
    supervoxel_to_graphene_blockwise(**relevant_args)

