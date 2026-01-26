import argparse
import json
import logging
import neuroglancer
import numpy as np

from cloudvolume import CloudVolume, Bbox
from funlib.segment.arrays import replace_values
from funlib.persistence import open_ds
from funlib.geometry import Roi
from pychunkedgraph.io.components import get_chunk_components
from pychunkedgraph.io.edges import get_chunk_edges
from time import sleep
from tqdm import tqdm

from segtopcg.utils.utils_supervoxel import get_chunk_coord


logging.basicConfig(level=logging.INFO)


def test(config, n_trials=1, view_seg=False):

    with open(config, 'r') as f:
        config = json.load(f)

    datapath = config['cloudpath']
    fragments = open_ds(config['fragments_file'], 'frags')
    chunk_size = np.array(config['chunk_voxel_size'][::-1])

    cv = CloudVolume(datapath + '/fragments')
    n_chunks = np.array(cv.shape[:-1]) // chunk_size

    for _ in tqdm(range(n_trials)):
        # Random chunk coords
        main_chunk_coords = np.random.randint(n_chunks)
        start = main_chunk_coords * chunk_size + cv.bounds.minpt
        bbox = Bbox(start, 
                    start + chunk_size) 

        print(main_chunk_coords)

        roi = Roi((main_chunk_coords * chunk_size * np.array(cv.resolution))[::-1] + fragments.roi.get_begin(), 
                (chunk_size * np.array(cv.resolution))[::-1])

        print(get_chunk_coord(fragments, roi, roi.get_shape())[::-1])

        # Get all adjacent chunks to get components from other chunks that cross into this one
        adj_chunk_coords_list = np.empty([6,3], dtype=int)
        i=0
        for d in [-1,1]:
            for dim in range(3):
                diff = np.zeros([3], dtype=int)
                diff[dim] = d
                adj_chunk_coords_list[i,:] = main_chunk_coords + diff
                i += 1

        chunk_coords = np.vstack([main_chunk_coords, adj_chunk_coords_list])

        # Get components
        comps = []
        for chunk in chunk_coords:
            comps.append(get_chunk_components(datapath + '/components', chunk))

        # Get edges
        edges = get_chunk_edges(datapath + '/edges', chunk_coords)
        in_edges = edges['in'].get_pairs()
        bt_edges = edges['between'].get_pairs()
        cross_edges = edges['cross'].get_pairs()

        # Get array
        data = cv[bbox]
        data = np.array(data)
        data = data.squeeze()

        # Get unique IDs
        frag_ids_comps = np.unique(np.concatenate([list(c.keys()) for c in comps]))
        frag_ids_cv = np.unique(data)
        nodes = np.unique(np.concatenate([in_edges, bt_edges, cross_edges]))


        # CHECKS

        if np.isin(frag_ids_comps, nodes).all():
            print('All frag ids in components contained in edges')
        if np.isin(frag_ids_cv[frag_ids_cv>0], nodes).all():
            print('All frag ids in CV contained in edges')
        if np.isin(frag_ids_cv[frag_ids_cv>0], frag_ids_comps).all():
            print('All frag ids in CV contained in components')

        if view_seg:
            old = np.concatenate([list(c.keys()) for c in comps])
            new = np.concatenate([list(c.values()) for c in comps])+1

            labels = data.T.copy()

            labels = replace_values(labels.astype(np.int64), 
                                    old.astype(np.int64), 
                                    new.astype(np.int64), 
                                    inplace=False)

            neuroglancer.set_server_bind_address('0.0.0.0', bind_port=None)
            dimensions = neuroglancer.CoordinateSpace(names=['z', 'y', 'x'], units='nm', scales=[50, 10, 10])
            viewer = neuroglancer.Viewer()
            with viewer.txn() as s:
                s.dimensions = dimensions    
                s.layers['og'] = neuroglancer.SegmentationLayer(source=neuroglancer.LocalVolume(data.T, 
                                                                                                dimensions))
                s.layers['labels'] = neuroglancer.SegmentationLayer(source=neuroglancer.LocalVolume(labels.astype(np.uint64), 
                                                                                                    dimensions))

            url = viewer.get_viewer_url()
            print('http://localhost:' + url.split(':')[-1])

            sleep(1)

            input()

if __name__ == '__main__':

    parser=argparse.ArgumentParser('')
    parser.add_argument('-cfg', '--config',
                        metavar='CONFIG',
                        dest='config',
                        required=True,
                        type=str,
                        help='Config used to upload data.')
    parser.add_argument('-n', '--n_trials',
                        metavar='N_TRIALS',
                        dest='n_trials',
                        required=False,
                        default=1,
                        type=int,
                        help='Number of random chunks to test')   
    parser.add_argument('-v', '--view_seg',
                        dest='view_seg',
                        action='store_true',
                        help='Whether to view the chunk with neuroglancer')
    args=parser.parse_args()

    test(**vars(args))