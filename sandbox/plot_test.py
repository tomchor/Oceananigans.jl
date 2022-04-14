import xarray as xr
from matplotlib import pyplot as plt
from xmovie import Movie
import pynanigans as pn
from dask.diagnostics import ProgressBar

#++++
var = "νₑ"
fname = "vid.csi_AMD3.nc"
sufix = fname.replace(".nc","").replace("vid.", "")

lim = 4e-3
vmin = 0
vmax = +lim
#----

#+++++ Open and rechunk
vid = xr.open_dataset(fname, decode_times=False)
vid = vid.chunk(dict(time=1))
#----

da = vid[var].pnisel(x=0)
mov = Movie(da, vmin=vmin, vmax=vmax)
with ProgressBar():
    mov.save(f"{var}_{sufix}.mp4", 
             parallel=True, 
             parallel_compute_kwargs=dict(), 
             overwrite_existing=True,
             )

