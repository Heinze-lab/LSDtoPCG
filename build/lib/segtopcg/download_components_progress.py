import argparse
import logging
import numpy as np
import pymongo

from caveclient import CAVEclient
from collections import defaultdict
from datetime import date

from segtopcg.utils.utils_components import get_adjacent_chunk_coords
from segtopcg.utils.utils_supervoxel import get_chunkId


logging.basicConfig(level=logging.INFO)


def get_chunk_ids_from_node_ids(segmentation_info, ids):

    '''
    Modified from Pychunkedgraph
    '''

    bits_per_dims = segmentation_info['graph']['spatial_bit_masks']['1']
    layer_id_bits = segmentation_info['graph']['n_bits_for_layer_id']

    offsets = 64 - layer_id_bits - 3 * bits_per_dims

    cids1 = np.array((np.array(ids, dtype=int) >> offsets) << offsets, dtype=np.uint64)
    return cids1


def get_chunk_coordinates(segmentation_info, node_or_chunk_id):

    '''
    Modified from Pychunkedgraph.
    '''

    bits_per_dims = segmentation_info['graph']['spatial_bit_masks']['1']
    layer_id_bits = segmentation_info['graph']['n_bits_for_layer_id']

    x_offset = 64 - layer_id_bits - bits_per_dims
    y_offset = x_offset - bits_per_dims
    z_offset = y_offset - bits_per_dims

    x = int(node_or_chunk_id) >> x_offset & 2 ** bits_per_dims - 1
    y = int(node_or_chunk_id) >> y_offset & 2 ** bits_per_dims - 1
    z = int(node_or_chunk_id) >> z_offset & 2 ** bits_per_dims - 1
    return np.array([x, y, z])


def get_adjacent_chunk_ids(segmentation_info, chunk_id):

    bits_per_chunk_dim = segmentation_info['graph']['spatial_bit_masks']['1']
    main_chunk = get_chunk_coordinates(segmentation_info, chunk_id)[::-1] # returns xyz
    adj_chunk_coords = get_adjacent_chunk_coords(main_chunk, [], 1)
    return [get_chunkId(bits_per_chunk_dim, chunk_coord=c) for c in adj_chunk_coords]


def download_progress_components(datastack_name,
                                 server_address,
                                 db_name,
                                 coll_name='info_progress_save',
                                 db_host=None
                                 ):

    logging.info(f'Downloading progress from {datastack_name} to db: {db_name}')
    client = CAVEclient(datastack_name=datastack_name, 
                    server_address=server_address)
    cg = client.chunkedgraph

    ts = cg.get_oldest_timestamp()

    # Get all the root IDs for neurons that have changed
    _, new_roots = cg.get_delta_roots(ts)
    new_roots = cg.get_latest_roots(new_roots)
    logging.info(f'Found {len(new_roots)} roots modified since {ts.date()}')

    # Get corresponding supervoxels
    leaves = cg.get_leaves_many(new_roots)

    # Sort components into dict chunk_id > components
    logging.info('Sorting into components...')
    components = defaultdict(list)
    dumped = []
    saved = []
    for root, component in leaves.items():
        chunk_ids = get_chunk_ids_from_node_ids(cg.segmentation_info, component)
        u_chunk_ids = np.unique(chunk_ids)

        if len(u_chunk_ids) == 1:
            # Dump any object that was not proofread across chunks
            dumped.append(root)
            continue
        else:
            saved.append(root)

        for chunk_id in u_chunk_ids:
            # Keep components including neighbors to keep full objects
            adj_chunk_ids = get_adjacent_chunk_ids(cg.segmentation_info, chunk_id)
            adj_chunk_ids += [chunk_id]
            components[chunk_id].append(component[np.isin(chunk_ids, adj_chunk_ids)].tolist())
    logging.info(f'Dumped {len(dumped)}/{len(new_roots)} objects because they did not cross chunk boundaries')

    # Insert in database
    mongoclient = pymongo.MongoClient(db_host)
    db = mongoclient[db_name]
    coll = db[coll_name]

    for chunk_id, comp in components.items():
        doc = {'chunk_id': int(chunk_id),
            'components': comp}
        coll.insert_one(doc)

    info_doc = {'datastack_name': datastack_name,
                'server_address': server_address,
                'saved_roots': list(map(int, saved)),
                'dumped_roots': list(map(int, dumped)),
                'timestamp': ts,
                'date': date.today().strftime('%d%m%Y')}
    coll.insert_one(info_doc)
    logging.info(f'Progress was saved in db: {db_name}')

if __name__ == '__main__':
    
    parser=argparse.ArgumentParser('')
    parser.add_argument('-d', '--datastack',
                        metavar='DATASTACK_NAME',
                        dest='datastack_name',
                        required=True,
                        type=str,
                        help='Name of the datastack as shown on the CAVE main page.')
    parser.add_argument('-s', '--server-address',
                        metavar='SERVER_ADDRESS',
                        dest='server_address',
                        required=True,
                        type=str,
                        help='Address of the server hosting the CAVE instance.')
    parser.add_argument('-db', '--db-name',
                        metavar='DB_NAME',
                        dest='db_name',
                        required=True,
                        type=str,
                        help='Name of the Mongo database where to store the progress data.')
    
    parser.add_argument('-c', '--coll-name',
                        metavar='COLL_NAME',
                        dest='coll_name',
                        default='info_progress_save',
                        type=str,
                        help='Name of the collection where to save the progress within db_name.')
    parser.add_argument('--db-host',
                        metavar='DB_HOST',
                        dest='db_host',
                        default=None,
                        type=str,
                        help='URI of the MongoDB where to store progress.')
    args=parser.parse_args()
    
    download_progress_components(datastack_name=args.datastack_name,
                                 server_address=args.server_address,
                                 db_name=args.db_name,
                                 coll_name=args.coll_name,
                                 db_host=args.db_host
                                 )