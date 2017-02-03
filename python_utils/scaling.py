import numpy as np

def scaling(inscale, indata, outscale=None, outdata=None, do_out=True):
    indata_min = inscale[0,:]
    indata_max = inscale[1,:]
    indata-=  indata_min
    indata /=  (indata_max-indata_min)
    indata *= 0.8
    indata += 0.1
    indata = indata.astype("float32")  
    if do_out:
        outdata_min = outscale[0,:]
        outdata_max = outscale[1,:]
        outdata = outdata - outdata_min
        outdata = outdata / (outdata_max-outdata_min)
        outdata =  (outdata * 0.8) + 0.1
        outdata = outdata.astype("float32") 
        return indata, outdata
    return indata