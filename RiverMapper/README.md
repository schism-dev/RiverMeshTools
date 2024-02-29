# RiverMapper: DEM-based meshing aid for compound flooding 
![Watershed rivers](Intro.jpg?raw=true)

Note: this is not a mesh generator; instead, it only generates river arcs or polygons that can be fed to mesh generators.
The purpose of the tool is to avoid manual labor of delineating watershed rivers, especially in continental-scale compound flood studies.

## Installation 
Python 3.9 or above is recommended

```bash
pip install git+https://github.com/schism-dev/RiverMeshTools.git#subdirectory=RiverMapper
```

You may need to manually install gdal and/or mpi4py (e.g., conda install gdal or mamba install gdal) if there is an error related to it, then redo the above 'pip install' command.

## Sample applications
Download link:
http://ccrm.vims.edu/yinglong/feiye/Public/RiverMapper_Samples.tar

Use the command
```bash
tar xvf RiverMapper_Samples.tar
```
to extract files from RiverMapper_Samples.tar.

## Tutorial
See online documentation in the [SCHISM manual](https://schism-dev.github.io/schism/master/mesh-generation/meshing-for-compound-floods/generate-river-map.html) for more information.
