Given:
- RGB Frames
- Depth Frames
- Cam2World Extrinsics
- Camera Intrinsics

- Convert to a 3D point cloud with colors 
- Given an additional SDF with world2grid transformation, transform pointcloud to SDF space
- Run marching cubes on SDF
