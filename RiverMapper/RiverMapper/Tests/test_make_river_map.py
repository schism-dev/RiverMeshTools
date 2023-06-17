import unittest
import numpy as np
from make_river_map import snap_closeby_points_global
import pickle

class TestMakeRiverMap(unittest.TestCase):
    def test_snap_closeby_points_global(self):
        # pt_xyz = np.loadtxt('/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.1/pt_xyz.txt')

        # with open('/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.1/pt_xyz.pkl', 'wb') as file:
        #     pickle.dump(pt_xyz, file)
        # with open('/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.1/xyz_nsnap.pkl', 'wb') as file:
        #     pickle.dump([xyz, nsnap], file)

        with open('/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.1/pt_xyz.pkl', 'rb') as file:
            pt_xyz = pickle.load(file)
        with open('/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.1/xyz_nsnap.pkl', 'rb') as file:
            xyz0, nsnap0 = pickle.load(file)

        # test the function
        xyz, nsnap = snap_closeby_points_global(pt_xyz)

        # assert results are the same to the expected ones
        assert np.array_equal(xyz, xyz0)
        assert nsnap == nsnap0

        pass
        
if __name__ == '__main__':
    # with open('/sciclone/schism10/Hgrid_projects/STOFS3D-V6/v16.1/pt_xyz.pkl', 'rb') as file:
    #     pt_xyz = pickle.load(file)
    # xyz, nsnap = snap_closeby_points_global(pt_xyz)

    unittest.main()