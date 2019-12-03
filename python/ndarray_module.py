import numpy as np
import open3d as o3d

print "[DEBUG] test"


def PassArrayFromCToPython(Array):
    # Array = np.array(Array)
    print "Shape Of Array:", Array.shape
    print Array, type(Array)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(Array[0])
    cloud.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([cloud])


