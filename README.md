## RiverMeshTools

Tools used for aiding mesh generation especially for compound flooding studies.
1.  pyDEM: extract thalwegs from DEM tiles, see more details in [RiverMeshTools/pyDEM](https://github.com/schism-dev/RiverMeshTools/tree/main/pyDEM)
2.  RiverMapper: generate polylines based on thalwegs and DEM tiles to guide watershed river mesh generation, see more details in [RiverMeshTools/RiverMapper](https://github.com/schism-dev/RiverMeshTools/tree/main/RiverMapper)

Note:
These two components can be used independently. For instance, to extract a 1-D river network, only the first component is necessary. Furthermore, the second component accepts any reasonable representation of a 1-D river network, such as those derived from existing hydrological models, manually created, or extracted from digital elevation models (DEMs), without requiring strict adherence to the thalwegs.

---

Publicaton:
Ye, F., Cui, L., Zhang, Y., Wang, Z., Moghimi, S., Myers, E., Seroka, G., Zundel, A., Mani, S. and Kelley, J.G., 2023. A parallel Python-based tool for meshing watershed rivers at continental scale. Environmental Modelling & Software, 166, p.105731.

The paper is accessible here: [DOI](https://doi.org/10.1016/j.envsoft.2023.105731), and if you don't have subscription to Environmental Modeling & Software, here is an [unofficial version of the manuscript](https://ccrm.vims.edu/yinglong/feiye/Public/RiverMapper_images/rivermeshtool_revision2.pdf).
Some of the functionalities described in the paper may have been upgraded.

---

Refer to [this section](https://schism-dev.github.io/schism/master/mesh-generation/meshing-for-compound-floods/overview.html) of the SCHISM manual for the most up-to-date tutorial.


