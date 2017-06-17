import pypcd
import numpy as np
pc = pypcd.PointCloud.from_path('data/new_loam/00/pcd/00008.pcd')

data = pc.pc_data

dx = (data['x']/0.4).astype(int)
dy = (data['y']/0.4).astype(int)
dz = (data['z']/0.4).astype(int)

pcd = np.array([dx, dy, dz]).transpose()
keep = abs(pcd[:,0] < 64) * abs(pcd[:,1] < 64) * abs(pcd[:,2] < 8)

pcd_out = pcd[keep]
ocmap = np.zeros([128, 128, 16])
ocmap[pcd_out[:,0],pcd_out[:,1], pcd_out[:,2]]=1
