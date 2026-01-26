from segtopcg.utils.utils_local import read_chunk_edges_local

import json
import logging
import numpy as np
import sys
import time


logging.basicConfig(level=logging.INFO)


def get_chunk_coordinates(node_or_chunk_id: np.uint64, bits_per_chunk_dim, layer_bits=8) -> np.ndarray:
        '''
        Modified from PyChunkedGraph
        Extract X, Y and Z coordinate from Node ID or Chunk ID
        :param node_or_chunk_id: np.uint64
        :return: Tuple(int, int, int)
        '''
        x_offset = 64 - layer_bits - bits_per_chunk_dim
        y_offset = x_offset - bits_per_chunk_dim
        z_offset = y_offset - bits_per_chunk_dim

        x = int(node_or_chunk_id) >> x_offset & 2 ** bits_per_chunk_dim - 1
        y = int(node_or_chunk_id) >> y_offset & 2 ** bits_per_chunk_dim - 1
        z = int(node_or_chunk_id) >> z_offset & 2 ** bits_per_chunk_dim - 1

        return np.array([x, y, z])

    
def get_chunk_ids_from_node_ids(ids, bits_per_chunk_dim, layer_bits = 8) -> np.ndarray:
    ''' Extract Chunk IDs from Node IDs'''
    if len(ids) == 0:
        return np.array([], dtype=np.uint64)

    offsets = 64 - layer_bits - 3 * bits_per_chunk_dim
    cids1 = np.array((np.array(ids, dtype=int) >> offsets) << offsets, dtype=np.uint64)
    return cids1


def get_chunk_ids_from_coords(coords: np.ndarray, bits_per_chunk_dim, layer_bits = 8):
    result = np.zeros(len(coords), dtype=np.uint64)

    layer_offset = 64 - layer_bits
    x_offset = layer_offset - bits_per_chunk_dim
    y_offset = x_offset - bits_per_chunk_dim
    z_offset = y_offset - bits_per_chunk_dim
    coords = np.array(coords, dtype=np.uint64)
    
    result |= 1 << layer_offset
    result |= coords[:, 0] << x_offset
    result |= coords[:, 1] << y_offset
    result |= coords[:, 2] << z_offset
    return result


def get_adjacent_chunk_coords(chunk_coords, 
                              chunks_to_cut,
                              group_size=1):
    
    '''
    Produces list of possible chunks adjacent to the chunk being processed, which are not to be isolated. 
    If chunk is to be isolated, returns empty np.array.
    
    Args:
            
        chunk_coords ([3] list of ``int``):
        
            Coordinates of the chunk (XYZ) to process.
            
        chunks_to_cut ([n,3] list of ``int``):
        
            List of chunk coordinates to be isolated from neighbors.
            
        group_size (``int``):
        
            Number of chunks to group together in each dimension. Default to 1: chunks are not to be grouped with neighbors.
            The group will override chunks_to_cut.
    '''
    
    adj_chunk_coords_list = np.empty([6,3], dtype=int)

    i=0
    for d in [-1,1]:
        for dim in range(3):
            diff = np.zeros([3], dtype=int)
            diff[dim] = d
            adj_chunk_coords_list[i,:] = chunk_coords + diff
            i += 1

    adj_chunk_coords_list = adj_chunk_coords_list[np.all(adj_chunk_coords_list >= 0, 1)]
    
    if len(chunks_to_cut) == 0:
        return adj_chunk_coords_list
    
    chunk_groups = adj_chunk_coords_list // group_size

    # Is chunk to be isolated?
    if np.any(np.all(chunk_coords == np.array(chunks_to_cut), 1)):
        # If yes, simply isolate it from all chunks around
        mask = np.zeros(adj_chunk_coords_list.shape[0], dtype=bool)
    else:
        # If not isolate from any chunk that needs to be isolated...
        mask = np.array([np.logical_not(np.any(np.all(chunk == chunks_to_cut, 1))) for chunk in adj_chunk_coords_list])

    # Always keep all chunks in the same group connected to each other
    main_group = np.array(chunk_coords) // group_size
    mask = mask | np.all(chunk_groups == main_group, 1)
        
    return adj_chunk_coords_list[mask]
    

def get_main_chunk_edges(edges_dir, 
                         main_chunk_coords, 
                         adj_chunk_coords, 
                         bits_per_chunk_dim):
    '''
    Reads local edges for main chunk and adjacent chunks.
    Returns edges that cross into the main chunk, and their scores.
    
    Args:
    
        edges_dir (``str``):
        
            Local path to the directory containing edges in protobuf format.
            
        main_chunk_coords ([3] list of ``int``):
        
            Coordinates of the chunk (XYZ) to process.
            
        adj_chunk_coords ([n,3] list of ``int``):
        
            List of coordinates of the adjacent chunks (XYZ) to the one being processed.
        
        bits_per_chunk_dim (``int``):
            
            Number of bits used to encode chunk ID in graphene format.
    '''
    
    main_chunk_edges = read_chunk_edges_local(edges_dir, [main_chunk_coords])
    
    # Read edges for main chunk and adjacent chunks to consider
    edges = []
    scores = []    
    for e_type in ['in', 'between', 'cross']:
        edges.append(main_chunk_edges[e_type].get_pairs())
        scores.append(main_chunk_edges[e_type].affinities)
        
    if adj_chunk_coords.size:
        adj_chunk_edges = read_chunk_edges_local(edges_dir, adj_chunk_coords)
        for e_type in ['between', 'cross']:
            edges.append(adj_chunk_edges[e_type].get_pairs())
            scores.append(adj_chunk_edges[e_type].affinities)
        chunk_ids = get_chunk_ids_from_coords(np.concatenate([[main_chunk_coords], adj_chunk_coords]), bits_per_chunk_dim)
    else:
        chunk_ids = get_chunk_ids_from_coords([main_chunk_coords], bits_per_chunk_dim)

    edges = np.concatenate(edges, dtype=np.uint64)
    scores = np.concatenate(scores, dtype=np.float32)
    
    # Filter edges to keep only those within or crossing to the main chunk
    
    if edges.size:
        edges_chunks_ids = get_chunk_ids_from_node_ids(edges, bits_per_chunk_dim)
        mask = np.all(np.isin(edges_chunks_ids, chunk_ids),1)
        return edges[mask], scores[mask]
    else:
        return np.array([]), np.array([])
    

def check_block(coll, chunk_coord):
    return coll.count_documents({'graphene_chunk_coord': chunk_coord.tolist()}) >=1

def isolate_chunks(mode,
                   chunk_list,
                   db_host = '',
                   db_name = ''):
    
    '''
    Returns a list of chunks to be isolated during components computation.
    Args:
    
        mode (``str``):
        
            "all": All chunks will be isolated from each other.
            ... TO BE ADDED
                
        chunk_list ([n,3] list of tuples of ``int``):
        
            All chunks coordinates (XYZ) for this dataset. Will be checked against content of db if relevant.
            
        db_host (``str``):
        
            URI to the MongoDB containing information relevant to the isolation mode.
            
        db_name (``str``):
        
            MongoDB database containing information relevant to the isolation mode.

    '''
    
    
    if mode == 'all':
        # All chunks are to be isolated
        return chunk_list
    
    else:
        print('No other mode was implemented for isolation yet.')
        sys.exit()
