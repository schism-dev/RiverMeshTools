## pyDEM

Scripts used for extracting thalwegs from DEMs.

---

Installation (python 3.9 or above is recommended):
1. Utility library developed by Dr. Zhengui Wang:
     
```bash
pip install git+https://github.com/wzhengui/pylibs.git
```
(You may need to install mpi4py seperately if the pip commond failed)

2. pyDEM:

```bash
pip install git+https://github.com/schism-dev/schism.git#subdirectory=src/Utility/Grid_Scripts/Compound_flooding/pyDEM
```
(You may need to install GDAL manually if the pip command failed)

---

Download sample applications [here](http://ccrm.vims.edu/yinglong/feiye/Public/pyDEM_Samples.tar) and extract using the command

```bash
tar xvf pyDEM_Samples.tar
```

---

See [this section](https://schism-dev.github.io/schism/master/mesh-generation/meshing-for-compound-floods/extract-thalweg.html) in the SCHISM manual for the most up-to-date tutorial.
