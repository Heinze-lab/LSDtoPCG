import daisy
import json
import logging
import numpy as np
import pymaid
import os

from funlib.persistence import open_ds
from funlib.geometry import Coordinate, Roi
from pymongo import MongoClient

from segtopcg.utils.utils_supervoxel import get_chunk_coord

logging.basicConfig(level=logging.INFO)

def prepare_skeleton_LUTs(fragments_file, 
                          fragments_dataset,
                          project_name,
                          volume_names,
                          chunk_voxel_size,
                          skeleton_luts_dir,
                          db_host, 
                          db_name,
                          catmaid_credentials,
                          num_workers
                          ):
    
    with open(catmaid_credentials, 'r') as f:
        catmaid_credentials = json.load(f)

    client = pymaid.CatmaidInstance(server=catmaid_credentials['server'], 
                                    api_token=catmaid_credentials['token'],
                                    http_user=catmaid_credentials['http_user'],
                                    http_password=catmaid_credentials['http_password'],
                                    caching=False)
    pid = client.available_projects.loc[client.available_projects.title == project_name, 'id'].values[0]
    client.project_id = pid
    
    # Prepare skeletons
    catmaid_volumes = [pymaid.get_volume(v) for v in volume_names]
    skeletons = pymaid.get_neurons_in_volume(catmaid_volumes)
    skeletons = pymaid.get_neurons(skeletons)
    skeletons.prune_by_volume(catmaid_volumes, inplace=True)

    # Prepare relevant nodes
    fragments = open_ds(fragments_file, fragments_dataset)
    nodes = skeletons.nodes.copy()
    mask = np.all(np.array(fragments.roi.begin) < nodes[['z', 'y', 'x']], axis=1)
    mask &= np.all(np.array(fragments.roi.end) > nodes[['z', 'y', 'x']], axis=1)
    nodes = nodes[mask]

    # Start blockwise processing
    chunk_size = Coordinate(chunk_voxel_size) * fragments.voxel_size
    task = daisy.Task(task_id = 'prepare_skeletons',
                      total_roi = fragments.roi,
                      read_roi = Roi((0,0,0), chunk_size),
                      write_roi = Roi((0,0,0), chunk_size),
                      process_function = lambda block: save_skeletons_LUT(block,
                                                                          nodes, 
                                                                          fragments, 
                                                                          skeleton_luts_dir,
                                                                          db_host, 
                                                                          db_name),
                      num_workers = num_workers,
                      read_write_conflict = False,
                      max_retries = 2,
                      fit = 'shrink'
                     )
    daisy.run_blockwise([task])


def save_skeletons_LUT(block, nodes, fragments, skeleton_luts_dir, db_host, db_name):
    
    # Mongo stuff
    client = MongoClient(db_host)
    db = client[db_name]
    trs_coll = db['ids_to_graphene']

    chunk_coord = get_chunk_coord(fragments, block.read_roi, block.read_roi.shape) # z,y,x
    chunk_coord = list(map(int, chunk_coord))
    
    # Only keep nodes in the ROI
    site_nodes = []
    for row in nodes[['node_id', 'z', 'y', 'x', 'neuron']].to_numpy():
        node_id = row[0]
        coords = row[1:-1]
        skid = row[-1]

        if block.read_roi.contains(coords):
            site_nodes.append({'id': node_id, 'skid': skid, 'z': coords[0], 'y': coords[1], 'x': coords[2]})

    # Make LUTs
    site_fragment_lut, num_bg_sites = get_site_fragment_lut(fragments, site_nodes, block.read_roi)

    if site_fragment_lut is None:
        return

    # Add graphene IDs
    try:
        trs_dict = trs_coll.find_one({'chunk_coord': chunk_coord},
                                    {'_id':0, 
                                    'initial_ids':1, 
                                    'graphene_ids':1})
        trs_dict = dict(zip(trs_dict['initial_ids'], trs_dict['graphene_ids']))
    except TypeError as e:
        raise TypeError(f'Chunk might not exist in the database: {chunk_coord}') from e

    graphene_ids = [trs_dict[frag] for frag in site_fragment_lut[2]]
    site_fragment_lut = np.concatenate([site_fragment_lut, np.array(graphene_ids, dtype=np.uint64)[None, ...]], axis=0)

    # Store LUT
    block_lut_path = os.path.join(
        skeleton_luts_dir,
        '_'.join([str(c) for c in chunk_coord[::-1]])) # x,y,z

    np.savez_compressed(
        block_lut_path,
        site_fragment_lut = site_fragment_lut,
        num_bg_sites = num_bg_sites)


def get_site_fragment_lut(fragments, sites, roi):
    # Get the fragment IDs of all the sites that are contained in the given ROI

    sites = list(sites)

    if len(sites) == 0:
        logging.debug(f'No sites in {roi}, skipping')
        return None, None

    fragment_ids = np.array([
        fragments[Coordinate((site['z'], site['y'], site['x']))]
        for site in sites
    ])
    site_ids = np.array(
        [site['id'] for site in sites],
        dtype=np.uint64)
    skids = np.array(
        [site['skid'] for site in sites],
        dtype=np.uint64)

    fg_mask = fragment_ids != 0
    fragment_ids = fragment_ids[fg_mask]
    site_ids = site_ids[fg_mask]
    skids = skids[fg_mask]

    lut = np.array([site_ids, skids, fragment_ids])

    return lut, (fg_mask==0).sum()