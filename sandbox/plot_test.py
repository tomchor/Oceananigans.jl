import xarray as xr
from matplotlib import pyplot as plt
from xmovie import Movie
import pynanigans as pn
from dask.diagnostics import ProgressBar

#++++
var = "ω_x"
fname = "vid.csi_SMA.nc"
sufix = fname.replace(".nc","").replace("vid.", "")
#----

#+++++ Open and rechunk
vid = xr.open_dataset(fname, decode_times=False)
vid = vid.chunk(dict(time=1))
#----

da = vid[var].pnisel(x=0)
mov = Movie(da)
with ProgressBar():
    mov.save(f"{var}_{sufix}.mp4", 
             parallel=True, 
             parallel_compute_kwargs=dict(), 
             overwrite_existing=True,
             )

