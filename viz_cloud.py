import open3d as o3d

pcd = o3d.io.read_point_cloud(r".\vlm_query_imgs\hdibix_clustered.ply")
o3d.visualization.draw_geometries([pcd])