import daisy
import numpy as np    

from funlib.persistence import open_ds


def get_segId(frag_ids,
              fragments=None,
              chunk_roi=None):

    '''
    Provides segment IDs in graphene format, for a chunk based on the position of supervoxels.
    
    Args:
    
        frag_ids (`class:numpy.ndarray`):

            Array containing fragments/supervoxel data.
            
        fragments (`class:daisy.Array`):

            Array containing fragments/supervoxel data.
            
        chunk_roi (`class:daisy.Roi`):
            
            Region of interest covered by chunk. 
           
    Return:
        
        Array of old IDs.
        Array of translated IDs.
            
    '''
    
    if frag_ids is None and fragments is not None:
        # Get data in chunk, transpose zyx to xyz
        data = fragments.intersect(chunk_roi).to_ndarray()
        frag_ids = np.unique(data).astype(np.uint64)
        
    frag_ids = frag_ids[frag_ids>0]
        
    if np.all(frag_ids == 0):
        return np.array([0], dtype = np.uint64), np.array([0], dtype = np.uint64)
    
    frag_ids.sort()
    local_ids = np.linspace(1, frag_ids.shape[0], frag_ids.shape[0], dtype = np.uint64)
    
    return frag_ids, local_ids
    

def get_chunk_list(fragments, 
                   chunk_size):
    '''
    Compute list of all xyz chunk coordinates for a given dataset.
    
    Args:
    
        fragments (`class:daisy.Array`):

            Array containing fragments/supervoxel data.
            
        chunk_size ([3,0] array of ``int``):
            
            Chunk size in world units (ZYX). Required to compute chunk coordinates if none provided.
            
    Return:
        
        [n] list of [3] tuples containing possible chunk coordinates (XYZ).
        List of [3] containing max number of chunks in each dimension (XYZ).
            
    '''

    ds = open_ds(fragments, 'frags')
    chunk_shape = np.array(chunk_size) / np.array(ds.voxel_size)
    n_chunks = np.ceil(np.array(ds.shape)/chunk_shape).astype(int)

    chunk_list = []
    for ix, x in enumerate(range(n_chunks[2])):
        for iy, y in enumerate(range(n_chunks[1])):
            for iz, z in enumerate(range(n_chunks[0])):
                chunk_list.append((x,y,z))
    
    return chunk_list, n_chunks[::-1]


def get_chunk_coord(fragments, chunk_roi, chunk_size, total_roi = None):

    '''
    Compute chunk_coord in reference to number of chunks in the dataset, from a given array or ROI 
    
    Args:
    
        fragments (`class:daisy.Array`):

            Array containing fragments/supervoxel data.
            
        chunk_roi (`class:daisy.Roi`):
            
            Region of interest covered by chunk. 
            
        chunk_size ([3,0] array of ``int``):
            
            Chunk size in world units (ZYX).
            
        total_roi (`class:daisy.Roi`):
        
            Total region of interest covered by the dataset.
    
    Return:
        
        [1,3] list of coordinates (ZYX)
    '''
    
    if total_roi is None:
        ds_offset = fragments.roi.get_begin()
    else:
        ds_offset = total_roi.get_begin()

    chunk_offset = chunk_roi.get_begin()
    chunk_rel_offset = chunk_offset - ds_offset

    return list(np.array(chunk_rel_offset)//np.array(chunk_size))


def get_chunkId(bits_per_chunk_dim, fragments=None, chunk_roi=None, chunk_size=None, chunk_coord=None):

    '''
    Computes the chunk ID for a given block, based on graphene format.
    
    Args:
    
        bits_per_chunk_dim (``int``):
        
            Number of bits used to encode chunk ID.
    
        fragments (`class:daisy.Array`):
            
            Array containing fragments/supervoxel data. Required to compute chunk coordinates if none provided.
        
        chunk_roi (`class:daisy.Roi`):
            
            Region of interest covered by chunk. Required to compute chunk coordinates if none provided.
            
        chunk_size ([3,0] array of ``int``):
            
            Chunk size in world units (ZYX). Required to compute chunk coordinates if none provided.
            
        chunk_coord ([3,0] array of ``int``):
            
            Chunk coordinates in fragments (ZYX). Will be computed if not provided.
    
    Return:
        
        Graphene ID (``np.uint64``) containing layer and chunk ID.
    '''
    
    layer_id = 1
    
    if chunk_coord is None:
        chunk_coord = get_chunk_coord(fragments, chunk_roi, chunk_size) # z,y,x coordinates

    chunk_coord = np.array(chunk_coord, dtype=int)
    if chunk_coord.ndim > 1:
        z, y, x = chunk_coord.T
    else:
        z, y, x = chunk_coord

    # 64 bits total length - 8 bits layer id length
    layer_offset = 64 - 8
    x_offset = layer_offset - bits_per_chunk_dim
    y_offset = x_offset - bits_per_chunk_dim
    z_offset = y_offset - bits_per_chunk_dim
    
    # Return np.uint64 in graphene format (with XYZ)
    return np.uint64(layer_id << layer_offset | 
                     x << x_offset |
                     y << y_offset |
                     z << z_offset) 


def get_nbit_chunk_coord(fragments, chunk_size):
    
    '''
    Computes the number of bits necessary to encode the maximum number of chunk for all axis.
    
    Args:
    
        fragments (`class:daisy.Array`):
            
            Array containing fragments/supervoxel data
            
            
        chunk_size ([3,0] array of ``int``):
            
            Chunk size in world units (ZYX)
    
    Return:
        
        Number of bits (``int``) to encode maximum value of number of chunk per axis
    
    '''
    
    # Computes how many chunks fit in each dimensions
    chunk_shape = np.array(chunk_size) / np.array(fragments.voxel_size)
    
    div = np.array(fragments.shape)//np.array(chunk_shape)
    
    # Return max number of chunks that can be represented
    return int(max(div)).bit_length()
