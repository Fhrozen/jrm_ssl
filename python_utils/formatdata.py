import numpy as np

def formatdata(frames, bs_agl, indata, outdata=None, skip=1, tgt2=False, test_p=False):
    if test_p: 
        idx_start = 0
    else:
        idx_start = np.random.randint(skip)
    tmp_ln = (indata.shape[0]-frames+1)
    tmp_input = np.zeros((tmp_ln, indata.shape[1], indata.shape[2], frames))
    if outdata is not None:
        tmp_output = np.zeros((tmp_ln, 1, outdata.shape[1], frames))
    for in_dx in range (tmp_ln):
        for dx_2 in range (frames):
            tmp_input[in_dx, :,:,dx_2] = indata[in_dx+dx_2,] 
            if outdata is not None:
                tmp_output[in_dx, 0,:,dx_2] =  outdata[in_dx+dx_2,]
    array_idx = np.arange(idx_start,tmp_input.shape[0],skip)
    tmp_input = tmp_input[array_idx,] 
    tmp_target = np.ones((tmp_input.shape[0]), dtype=np.int32)*bs_agl

    if outdata is not None:
        tmp_output = tmp_output[idx_start:tmp_output.shape[0]:skip,] 
        if tgt2:
            _angle = np.pi*bs_agl/18
            eje_x = int(30*np.around(np.cos(_angle),decimals=3))
            eje_y = int(30*np.around(np.sin(_angle),decimals=3))

            tmp_target2 = np.zeros((tmp_input.shape[0],2,91,91))
            tmp_target2[:,1,44:47,44:47] = 0.8
            tmp_target2[:,0,44+eje_x:47+eje_x,44+eje_y:47+eje_y] = 1

        
            return tmp_input, tmp_output, tmp_target, tmp_target2
        return tmp_input, tmp_output, tmp_target
    if test_p: 
        return tmp_input, tmp_target, array_idx
    return tmp_input, tmp_target
