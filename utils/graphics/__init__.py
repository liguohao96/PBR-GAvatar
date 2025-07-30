import torch
import torch.nn.functional as F
import numpy as np

def rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    # K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def generate_cube(edge_length=1):
    cube = [-0.5, -0.5,  0.5, 1.0, 0.0, 0.0,
         0.5, -0.5,  0.5, 0.0, 1.0, 0.0,
         0.5,  0.5,  0.5, 0.0, 0.0, 1.0,
        -0.5,  0.5,  0.5, 1.0, 1.0, 1.0,

        -0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
         0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
         0.5,  0.5, -0.5, 0.0, 0.0, 1.0,
        -0.5,  0.5, -0.5, 1.0, 1.0, 1.0]

    cube = np.array(cube, dtype = np.float32).reshape(-1, 6)
    indices = [0, 1, 2, 2, 3, 0,
            4, 6, 5, 6, 4, 7,
            4, 5, 1, 1, 0, 4,
            6, 7, 3, 3, 2, 6,
            5, 6, 2, 2, 1, 5,
            7, 4, 0, 0, 3, 7]
    indices = np.array(indices, dtype= np.uint32).reshape(-1, 3)

    ver = cube[:, :3] * (edge_length)
    tri = indices
    return ver, tri

def face_normal(ver, tri):
    BS  = ver.size(0)

    faces   = ver[:, tri.flatten().long(), :].reshape(BS, -1, 3, 3)
    ori_face_nrm = torch.cross(faces[:,:,1]-faces[:,:,0], faces[:,:,2]-faces[:,:,0], dim=-1)
    return F.normalize(ori_face_nrm, dim=-1)

def vertex_normal(ver, tri):
    BS  = ver.size(0)

    tri = tri.long()

    ori_face_nrm = face_normal(ver, tri)

    v_nrm = torch.zeros_like(ver)
    v_nrm.scatter_add_(1, tri[None, :, 0, None].expand(BS, -1, 3), ori_face_nrm)
    v_nrm.scatter_add_(1, tri[None, :, 1, None].expand(BS, -1, 3), ori_face_nrm)
    v_nrm.scatter_add_(1, tri[None, :, 2, None].expand(BS, -1, 3), ori_face_nrm)
    return F.normalize(v_nrm, dim=-1)