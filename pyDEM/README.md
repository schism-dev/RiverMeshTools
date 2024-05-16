## pyDEM

This tool extracts thalwegs from DEMs through the following steps: 1) Filling depression, 2) Calculating flow direction, 3) Calculating flow accumulation, and finally select the DEM cells with flow accumulations larger than a user-specified threshold.

This first step depends on [richdem](https://pypi.org/project/richdem/) to fill depression. The algorithm used is "Priority-Flood+Epsilon" from 
"C Barnes, R., Lehman, C., Mulla, D., 2014. Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. Computers & Geosciences 62, 117–127." [doi:10.1016/j.cageo.2013.04.024](https://doi.org/10.1016/j.cageo.2013.04.024).

---

Installation (python 3.9 or 3.10 is recommended):
1. Utility library developed by Dr. Zhengui Wang:
     
```bash
pip install git+https://github.com/wzhengui/pylibs.git
```
(You may need to install mpi4py seperately if the pip commond failed)

2. pyDEM:

```bash
pip install git+https://github.com/schism-dev/RiverMeshTools.git#subdirectory=pyDEM
```
(You may need to install GDAL manually if the pip command failed)

---

Download sample applications [here](http://ccrm.vims.edu/yinglong/feiye/Public/pyDEM_Samples.tar) and extract using the command

```bash
tar xvf pyDEM_Samples.tar
```

---

See [this section](https://schism-dev.github.io/schism/master/mesh-generation/meshing-for-compound-floods/extract-thalweg.html) in the SCHISM manual for the most up-to-date tutorial.
