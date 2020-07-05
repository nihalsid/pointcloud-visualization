import struct
import numpy as np
import torch
from pathlib import Path
import util.marching_cubes.marching_cubes as mc
from imageio import imread


def points_to_obj(points, colors, path):
    with open(path, 'w') as f:
        for i in range(points.shape[0]):
            v = points[i, :]
            c = colors[i, :]
            if not (v[0] == 0 and v[1] == 0 and v[2] == 0):
                f.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))


def sparse_to_dense_np(locs, values, dimx, dimy, dimz, default_val):
    nf_values = 1 if len(values.shape) == 1 else values.shape[1]
    dense = np.zeros([dimz, dimy, dimx, nf_values], dtype=values.dtype)
    dense.fill(default_val)
    dense[locs[:, 0], locs[:, 1], locs[:, 2], :] = values
    if nf_values > 1:
        return dense.reshape([dimz, dimy, dimx, nf_values])
    return dense.reshape([dimz, dimy, dimx])


def read_sdf_header_only(sdf_file_path):
    fin = open(sdf_file_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    fin.close()
    return dimx, dimy, dimz, voxelsize, np.array(world2grid).astype(np.float32).reshape((4, 4))


def read_sdf(sdf_file_path):
    fin = open(sdf_file_path, 'rb')
    dimx = struct.unpack('Q', fin.read(8))[0]
    dimy = struct.unpack('Q', fin.read(8))[0]
    dimz = struct.unpack('Q', fin.read(8))[0]
    voxelsize = struct.unpack('f', fin.read(4))[0]
    world2grid = struct.unpack('f' * 4 * 4, fin.read(4 * 4 * 4))
    # input data
    num = struct.unpack('Q', fin.read(8))[0]
    input_locs = struct.unpack('I' * num * 3, fin.read(num * 3 * 4))
    input_locs = np.asarray(input_locs, dtype=np.int32).reshape([num, 3])
    input_locs = np.flip(input_locs, 1).copy()  # convert to zyx ordering
    input_sdfs = struct.unpack('f' * num, fin.read(num * 4))
    input_sdfs = np.asarray(input_sdfs, dtype=np.float32)
    input_sdfs /= voxelsize
    input_sdfs = sparse_to_dense_np(input_locs, input_sdfs[:, np.newaxis], dimx, dimy, dimz, -float('inf'))
    fin.close()
    return input_sdfs, dimx, dimy, dimz, np.array(world2grid).astype(np.float32).reshape((4, 4))


def convert_sdf_to_mesh(sdf_file_path, output_mesh_path):
    sdf, dimx, dimy, dimz, _ = read_sdf(sdf_file_path)
    print(dimx, dimy, dimz)
    mc.marching_cubes(torch.from_numpy(sdf), None, isovalue=0, truncation=3, thresh=10, output_filename=output_mesh_path)


def read_camera(camera_path):
    intrinsics, extrinsics = None, None
    with open(camera_path, "r") as fptr:
        all_lines = fptr.read().splitlines()
        extrinsic_elems = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in all_lines[:4])]
        intrinsic_elems = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in all_lines[4:])]
        extrinsics = np.asarray(extrinsic_elems, dtype=np.float32)
        intrinsics = np.asarray(intrinsic_elems, dtype=np.float32)
    return intrinsics, extrinsics


def project_rgbd_list_to_point_cloud(sdf_path, frame_list_path, frames_directory):
    list_of_frames = [x.strip() for x in Path(frame_list_path).read_text().splitlines() if x.strip() != ""]
    color, depth, intrinsic, extrinsic = [], [], [], []
    all_3d_locs = []
    all_3d_colors = []
    _, _, _, _, world2grid = read_sdf_header_only(sdf_path)
    for i, f in enumerate(list_of_frames):
        color_path = Path(frames_directory) / "color" / f"{f}.jpg"
        depth_path = Path(frames_directory) / "depth" / f"{f}.png"
        camera_path = Path(frames_directory) / "camera" / f"{f}.txt"
        color_image = imread(color_path).astype(np.float32)
        print(color_path)
        depth_image = imread(depth_path).astype(np.float32) / 1000
        intrinsic_mat, extrinsic_mat = read_camera(camera_path)
        color.append(color_image)
        depth.append(depth_image)
        intrinsic.append(intrinsic_mat)
        extrinsic.append(extrinsic_mat)

    for i in range(len(color)):
        cam_w = torch.from_numpy(np.matmul(world2grid, extrinsic[i]))
        cam_k = torch.from_numpy(intrinsic[i])
        x = torch.arange(0, depth[i].shape[1])
        y = torch.arange(0, depth[i].shape[0])
        y, x = torch.meshgrid(y, x)
        x = x.flatten().float()
        y = y.flatten().float()
        flat_depth = torch.from_numpy(depth[i]).flatten()
        flat_colors = color[i][y.int(), x.int()]
        image = torch.zeros(4, flat_depth.shape[0])
        image[0, :] = x * flat_depth
        image[1, :] = y * flat_depth
        image[2, :] = flat_depth
        image[3, :] = 1.0
        loc3d = torch.mm(cam_w, torch.mm(torch.inverse(cam_k), image)).T.numpy()
        all_3d_locs.append(loc3d)
        all_3d_colors.append(flat_colors)

    g_pointcloud = np.concatenate(all_3d_locs, axis=0)
    c_pointcloud = np.concatenate(all_3d_colors, axis=0)
    print(g_pointcloud.shape)
    points_to_obj(g_pointcloud, c_pointcloud, "test_pts.obj")


if __name__ == "__main__":
    project_rgbd_list_to_point_cloud("main/data/2t7WUuJeko7_room0__cmp__0.sdf", "main/data/2t7WUuJeko7_room0__cmp__0.txt", "main/data/frames_2t7WUuJeko7")
    convert_sdf_to_mesh("main/data/2t7WUuJeko7_room0__cmp__0.sdf", "test_mesh.obj")
