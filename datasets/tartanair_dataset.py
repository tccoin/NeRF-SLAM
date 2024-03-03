import glob
import os
import json
import numpy as np

import cv2
from tqdm import tqdm

from icecream import ic
from datasets.dataset import Dataset, Resolution, PinholeCameraModel, RadTanDistortionModel, CameraCalibration

class TartanairDataset(Dataset):
    def __init__(self, args, device):
        super().__init__("Tartanair", args, device)
        self.dataset_dir = args.dataset_dir
        self.parse_dataset()
        self._build_dataset_index()

    def load_poses(self, path):
        poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        f = open(f'{self.args.output}/gt_traj.txt', 'w')
        for i, line in enumerate(lines):
            data = line.split()
            t = np.array([float(x) for x in data[:3]])
            q = np.array([float(x) for x in data[3:]])
            c2w = np.eye(4)
            import scipy.spatial.transform
            c2w[:3,:3] = scipy.spatial.transform.Rotation.from_quat(q).as_matrix()
            c2w[:3, 3] = t
            f.write(f'{i} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n')
            w2c = np.linalg.inv(c2w)
            poses.append(w2c)
        f.close()
        return poses

    def _get_cam_calib(self, path):
        with open(os.path.join(self.dataset_dir, path), 'r') as f:
            self.json = json.load(f)

        camera = self.json["camera"]
        w, h = camera['w'], camera['h']
        fx, fy, cx, cy= camera['fx'], camera['fy'], camera['cx'], camera['cy']

        k1, k2, p1, p2 = 0, 0, 0, 0
        body_T_cam0 = np.eye(4)
        rate_hz = 0

        resolution  = Resolution(w, h)
        pinhole0    = PinholeCameraModel(fx, fy, cx, cy)
        distortion0 = RadTanDistortionModel(k1, k2, p1, p2)

        aabb = np.array([[-2, -2, -2], [2, 2, 2]]) # Computed automatically in to_nerf()
        depth_scale = 1.0 / camera["scale"] # Since we multiply as gt_depth *= depth_scale, we need to invert camera["scale"]

        return CameraCalibration(body_T_cam0, pinhole0, distortion0, rate_hz, resolution, aabb, depth_scale)

    def parse_dataset(self):
        self.timestamps = []
        self.poses      = []
        self.images     = []
        self.depths     = []
        self.calibs     = []

        self.image_paths = sorted(glob.glob(f'{self.dataset_dir}/image_left/*.png')) #left
        self.depth_paths = sorted(glob.glob(f'{self.dataset_dir}/depth_left/*.npy'))
        self.poses       = self.load_poses(f'{self.dataset_dir}/pose_left.txt')
        self.calib       = self._get_cam_calib(f'{self.dataset_dir}/../../../../cam_params.json')

        # N = self.args.buffer
        H, W = self.calib.resolution.height, self.calib.resolution.width

        self.resize_images = False
        if self.calib.resolution.total() > 640*640:
            self.resize_images = True
            self.output_image_size = [640, 480] # h, w 

        if self.resize_images:
            h0, w0  = self.calib.resolution.height, self.calib.resolution.width
            total_output_pixels = (self.output_image_size[0] * self.output_image_size[1])
            self.h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.h1 = self.h1 - self.h1 % 8
            self.w1 = self.w1 - self.w1 % 8
            self.calib.camera_model.scale_intrinsics(self.w1 / w0, self.h1 / h0)
            self.calib.resolution = Resolution(self.w1, self.h1)

        subset_poses = []    

        if self.final_k == -1:
            self.final_k = len(self.poses) - 1

        # Parse images and tfs
        print(f'Loading {len(self.image_paths)} images and tfs')
        self.tqdm = tqdm(total=len(self.image_paths)/self.img_stride, desc='Loading dataset')
        for i, (image_path, depth_path) in enumerate(zip(self.image_paths, self.depth_paths)):

            if ((i-self.initial_k) % self.img_stride) != 0 or i < self.initial_k or i > self.final_k:
                continue

            self.tqdm.update(1)

            # Parse rgb/depth images
            image = cv2.imread(image_path)
            depth = np.load(depth_path)



            if self.resize_images:
                w1, h1 = self.w1, self.h1
                image = cv2.resize(image, (w1, h1))
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) # Required for Nerf Fusion, perhaps we can put it in there

                depth = cv2.resize(depth, (w1, h1))

                if self.viz:
                    cv2.imshow(f"Img Resized", image)
                    cv2.imshow(f"Depth Resized", depth)
                    cv2.waitKey(1)
    
            depth = depth[:, :, np.newaxis]

            H, W, _  = depth.shape

            assert(H == image.shape[0])
            assert(W == image.shape[1])
            assert(3 == image.shape[2] or 4 == image.shape[2])
            assert(np.uint8 == image.dtype)
            assert(1 == depth.shape[2])
            assert(np.float32 == depth.dtype)

            depth = depth.astype(np.int32) 

            self.timestamps += [i]
            self.images     += [image]
            self.depths     += [depth]
            subset_poses    += [self.poses[i]]

        self.tqdm.close()
        self.poses = subset_poses

        # poses_t = np.array([x[:3,3].flatten() for x in self.poses])
        # delta_t = 2.0
        # t_max = np.amax(poses_t, 0).flatten()
        # t_min = np.amin(poses_t, 0).flatten()
        # aabb = np.array([t_min-delta_t, t_max+delta_t]).tolist()
        aabb = np.array([[-5]*3, [5]*3]).tolist()
        self.calib.aabb = aabb
        ic(aabb)


        self.timestamps = np.array(self.timestamps)
        self.poses      = np.array(self.poses)
        self.images     = np.array(self.images)
        self.depths     = np.array(self.depths)
        self.calibs     = np.array([self.calib]*len(self.images))

        N = len(self.timestamps)
        assert(N == self.poses.shape[0])
        assert(N == self.images.shape[0])
        assert(N == self.depths.shape[0])
        assert(N == self.calibs.shape[0])

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, k):
        return self.data_packets[k] if self.data_packets is not None else self._get_data_packet(k)

    def _get_data_packet(self, k0, k1=None):
        if k1 is None:
            k1 = k0 + 1
        else:
            assert(k1 >= k0)
        return {"k":      np.arange(k0,k1),
                "t_cams": self.timestamps[k0:k1],
                "poses":  self.poses[k0:k1],
                "images": self.images[k0:k1],
                "depths": self.depths[k0:k1],
                "calibs": self.calibs[k0:k1],
                "is_last_frame": (k0 >= self.__len__() - 1),
                }

    def _build_dataset_index(self):
        self.data_packets = [data_packet for data_packet in self.stream()]

    def stream(self):
        for k in range(self.__len__()):
            yield self._get_data_packet(k)

    def to_nerf_format(self):
        print("Exporting Replica dataset to Nerf")
        OUT_PATH = "transforms.json"
        AABB_SCALE = 4
        out = {
            "fl_x": self.calib.camera_model.fx,
            "fl_y": self.calib.camera_model.fy,
            "k1": self.calib.distortion_model.k1,
            "k2": self.calib.distortion_model.k2,
            "p1": self.calib.distortion_model.p1,
            "p2": self.calib.distortion_model.p2,
            "cx": self.calib.camera_model.cx,
            "cy": self.calib.camera_model.cy,
            "w": self.calib.resolution.width,
            "h": self.calib.resolution.height,
            "aabb": self.calib.aabb,
            "aabb_scale": AABB_SCALE,
            "integer_depth_scale": self.calib.depth_scale,
            "frames": [],
        }

        poses_t = []
        if self.data_packets is None:
            self._build_dataset_index()
        for data_packet in self.data_packets:
            # Image
            ic(data_packet["k"])
            color_path = self.image_paths[data_packet["k"][0]]
            depth_path = self.depth_paths[data_packet["k"][0]]

            relative_color_path = os.path.join("results", os.path.basename(color_path))
            relative_depth_path = os.path.join("results", os.path.basename(depth_path))

            # Transform
            w2c = data_packet["poses"][0]
            c2w = np.linalg.inv(w2c)

            # Convert from opencv convention to nerf convention
            # c2w[0:3, 1] *= -1  # flip the y axis
            # c2w[0:3, 2] *= -1  # flip the z axis

            poses_t += [w2c[:3,3].flatten()]

            frame = {"file_path": relative_color_path,  # "sharpness": b,
                     "depth_path": relative_depth_path,
                     "transform_matrix": c2w.tolist()}
            out["frames"].append(frame)

        poses_t = np.array(poses_t)
        delta_t = 2.0 # 1 meter extra to allow for the depth of the camera
        t_max = np.amax(poses_t, 0).flatten()
        t_min = np.amin(poses_t, 0).flatten()
        out["aabb"] = np.array([t_min-delta_t, t_max+delta_t]).tolist()

        # Save the path to the ground-truth mesh as well
        out["gt_mesh"] = os.path.join("..", os.path.basename(self.dataset_dir)+"_mesh.ply")
        ic(out["gt_mesh"])

        with open(OUT_PATH, "w") as outfile:
            import json
            json.dump(out, outfile, indent=2)
