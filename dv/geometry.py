# System
import typing as ty

# Third Party
import torch as t


def affine_identity(dim: int = 3, batch_size: ty.Optional[int] = None) -> t.Tensor:
    """Construct the affine matrix in homogeneous coordinates for an identity transformation.

    Args:
        dim: Dimension of the transformation.
        batch_size: Number of copies to return.

    Returns:
        If `batch_size` is `None`, dimension will be `[dim, (dim + 1)]`.
        If `batch_size` is `B`, dimension will be `[B, dim, (dim + 1)]`.

    Examples:
        >>> affine_identity()
        tensor([[1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.]])

        >>> affine_identity(dim=2, batch_size=2)
        tensor([[[1., 0., 0.],
                 [0., 1., 0.]],
        <BLANKLINE>
                [[1., 0., 0.],
                 [0., 1., 0.]]])
    """
    assert dim >= 2
    I = t.cat((t.ones((dim)).diag(), t.zeros(dim).unsqueeze(1)), dim=1)
    if batch_size is not None:
        I = I.unsqueeze(0).repeat(batch_size, 1, 1)
    return I


def batchwise_dot(v1: t.Tensor, v2: t.Tensor) -> t.Tensor:
    """Compute the dot product of two vectors or two batches of vectors
    Args:
        v1: Vector of shape `[N]` or batch of vectors of shape `[B, N]`.
        v2: Vector of shape `[N]` or batch of vectors of shape `[B, N]`.

    Returns:
        If `v1` and `v2` have shape `[N]`, the returned vector has shape `[1]`.
        If `v1` and `v2` have shape `[B, N]`, the returned vector has shape `[B, 1]`.

    Examples:
        >>> x = t.Tensor([1.0, 2.0, 3.0, 4.0])
        >>> y = t.Tensor([1.0, 2.0, 3.0, 4.0])
        >>> batchwise_dot(x, y)
        tensor([30.])

        >>> x = x.repeat(3, 1)
        >>> y = y.repeat(3, 1)
        >>> batchwise_dot(x, y)
        tensor([[30.],
                [30.],
                [30.]])
    """
    if v1.dim() == v2.dim() == 1:
        return t.Tensor([t.dot(v1, v2)])
    elif v1.dim() == v2.dim() == 2:
        v1 = v1.unsqueeze(1)
        v2 = v2.unsqueeze(-1)
        return t.bmm(v1, v2).squeeze(1)
    else:
        assert False


def angle(v1: t.Tensor, v2: t.Tensor) -> t.Tensor:
    """Compute the angle between two vectors, or between two batches of vectors.

    Args:
        v1: Vector of shape `[N]` or batch of vectors of shape `[B, N]`.
        v2: Vector of shape `[N]` or batch of vectors of shape `[B, N]`.

    Returns:
        The angle or batch of angles between `v1` and `v2`.
        If `v1` and `v2` have shape `[N]`, the returned vector has shape `[1]`.
        If `v1` and `v2` have shape `[B, N]`, the returned vector has shape `[B, 1]`.

    Examples:
        >>> x = t.Tensor([1.0, 0.0, 0.0])
        >>> y = t.Tensor([-1.0, 0.0, 0.0])
        >>> angle(x, x)
        tensor([0.])

        >>> angle(x, y)
        tensor([3.1416])

        >>> x = x.repeat(4, 1)
        >>> y = y.repeat(4, 1)
        >>> angle(x, y)
        tensor([[3.1416],
                [3.1416],
                [3.1416],
                [3.1416]])
    """
    if v1.dim() == v2.dim() == 1:
        return t.Tensor(
            [
                t.acos(
                    t.clamp(
                        t.dot(v1, v2) / (t.linalg.norm(v1) * t.linalg.norm(v2)),
                        min=-1.0,
                        max=1.0,
                    )
                )
            ]
        )
    elif v1.dim() == v2.dim() == 2:
        return t.arccos(
            t.clamp(
                batchwise_dot(v1, v2)
                / (t.linalg.norm(v1, dim=1) * t.linalg.norm(v2, dim=1)).unsqueeze(1),
                min=-1.0,
                max=1.0,
            )
        )
    else:
        assert False, "Expected tensors of 1 or 2 dimensions."


def axis_angle_to_quaternion(
    axis: t.Tensor, alpha: t.Tensor, ordering="xyzw"
) -> t.Tensor:
    """Convert rotation from an axis-angle to a quaternion representation.

    Args:
        axis: Vector or batch of vectors around which the rotation is to be taken; should have shape `[3]` or `[B, 3]`.
        alpha: Angle or batch of angles of rotation (radians); should have shape `[1]` or `[B, 1]`.

    Returns:
        If a single axis and angle are provided, return a quaternion of shape `[4]`.
        If a batch of axes and angles are provided, return a quaternion of shape `[B, 4]`.

    Examples:
        >>> axis = t.Tensor([1, 0, 0])
        >>> angle = t.Tensor([0.0])
        >>> axis_angle_to_quaternion(axis, angle)
        tensor([0., 0., 0., 1.])

        >>> axis_angle_to_quaternion(axis, angle, ordering="wxyz")
        tensor([1., 0., 0., 0.])

        >>> axis = axis.repeat(3, 1)
        >>> angle = angle.repeat(3, 1)
        >>> axis_angle_to_quaternion(axis, angle)
        tensor([[0., 0., 0., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 1.]])

        >>> axis_angle_to_quaternion(axis, angle, ordering="wxyz")
        tensor([[1., 0., 0., 0.],
                [1., 0., 0., 0.],
                [1., 0., 0., 0.]])
    """
    alpha_half = alpha / 2.0
    s = t.sin(alpha_half)
    c = t.cos(alpha_half)
    xyz = axis * s

    if ordering == "xyzw":
        return t.cat((xyz, c), -1)
    elif ordering == "wxyz":
        return t.cat((c, xyz), -1)
    else:
        print("Ordering not recognized! Allowed options are 'xyzw' or 'wxyz'.")
        print("'xyzw' is used as default for consistency with the RoMa library.")
        assert False


def major_axis(points: t.Tensor, align_with: t.Tensor) -> t.Tensor:
    """Given a pointset or batch of pointsets, use principal components analysis (PCA) to determine the primary axis.

    Args:
        points: Pointset of shape `[N, D]`, where `N` is the number of samples and `D` is the dimension of the data; optionally, a batch of coordinates shape `[B, N, D]` may be provided.
        align_with: Vector of shape `[D]` or batch of bectors of shape `[B, D]` along which the major axis should be re-aligned. That is, if the angle between the major axis and the provided vector is greater than 180 degrees, the returned vector is reversed.

    Returns:
        A vector of shape `[D]` or batch of vectors of shape `[B, D]` representing the major axis or axes of the provided pointset(s).

    Examples:
        >>> p = t.Tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        >>> e = t.Tensor([1, 0, 0])
        >>> major_axis(p, e)
        tensor([1., 0., 0.], dtype=torch.float16)

        >>> major_axis(p, -e)
        tensor([-1., -0., -0.], dtype=torch.float16)

        >>> p = t.stack((p, p))
        >>> e = t.stack((e, -e))
        >>> major_axis(p, e)
        tensor([[ 1.,  0.,  0.],
                [-1., -0., -0.]], dtype=torch.float16)
    """
    assert points.dim() == align_with.dim() + 1
    assert points.dim() in {2, 3}
    align_with = align_with.half().to(points.device)
    pca = t.pca_lowrank(points.double(), center=True)[2]
    V0 = pca[..., 0].half()
    flip = (batchwise_dot(V0, align_with) < 0.0).long() * -2 + 1
    V0 *= flip
    if points.dim() == 2:
        return V0.squeeze(0).squeeze(0)
    elif points.dim() == 3:
        return V0.squeeze(1)


def normalizing_quaternion_from_points(points: t.Tensor, newaxis: t.Tensor) -> t.Tensor:
    """Find the rotation quaternion which aligns the major axis of a pointset with a new axis. Batch processing also supported.

    Args:
        points: Points to be aligned, shape `[N, 3]`, or a batch of points shape `[B, N, 3]`.
        newaxis: Axis to which points are to be aligned, shape `[3]`, or a batch of axes of shape `[B, 3]`.

    Returns:
        A dictionary containing the rotating quaternion ("quaternion"), the major axis of the provided points ("axis"), and the angle between the major axis and the provided `newaxis` ("angle").

    Examples:

        >>> p = t.Tensor([[0, 0, 0], [1, 1, 0], [2, 2, 0], [3, 3, 0]])
        >>> n = t.Tensor([1, 0, 0])
        >>> q = normalizing_quaternion_from_points(p, n)
        >>> q['quaternion']
        tensor([0.0000, 0.0000, 0.3828, 0.9238])
        >>> t.arccos(q['quaternion'][-1])*2*180/t.pi
        tensor(45.0141)
        >>> q['quaternion'][:3] / t.sin(t.arccos(q['quaternion'][-1]))
        tensor([0.0000, 0.0000, 1.0000])

        >>> p = p.unsqueeze(0).repeat(2, 1, 1)
        >>> n = n.unsqueeze(0).repeat(2, 1)
        >>> q = normalizing_quaternion_from_points(p, n)
        >>> q['quaternion']
        tensor([[0.0000, 0.0000, 0.3828, 0.9238],
                [0.0000, 0.0000, 0.3828, 0.9238]], dtype=torch.float16)
        >>> t.arccos(q['quaternion'][:,-1])*2*180/t.pi
        tensor([45., 45.], dtype=torch.float16)
        >>> q['quaternion'][:,:3] / t.sin(t.arccos(q['quaternion'][:,-1].unsqueeze(1)))
        tensor([[0., 0., 1.],
                [0., 0., 1.]], dtype=torch.float16)

        >>> p = t.Tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        >>> n = t.Tensor([1, 0, 0])
        >>> q = normalizing_quaternion_from_points(p, n)
        >>> q['quaternion']
        tensor([0., 0., 0., 1.])

        >>> p = t.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        >>> n = t.Tensor([1, 0, 0])
        >>> q = normalizing_quaternion_from_points(p, n)
        >>> q['quaternion']
        tensor([0., 0., 0., 1.])
    """
    V = major_axis(points, newaxis)
    alpha = angle(V, newaxis.half())

    Vp = t.cross(newaxis.float(), V.float(), dim=-1).half()
    Vp = Vp / t.Tensor(t.linalg.norm(Vp, dim=-1, keepdim=True))
    return_dict = dict()
    return_dict["quaternion"] = axis_angle_to_quaternion(Vp, alpha).nan_to_num()
    return_dict["axis"] = V
    return_dict["angle"] = alpha
    return return_dict


def transform_from_seg(seg: t.Tensor, e: t.Tensor):
    """Determine the transformation required to align the major axis of `seg` with the provided tensor `e`.

    Args:
        seg: Segmentation.
        e: Vector to align with.

    Returns:
        A dictionary containing the rotating quaternion ("quaternion"), the major axis of the provided points ("axis"), the angle between the major axis and the provided `newaxis` ("angle"), and the centroid of the points ("centroid").

    Examples:
        >>> seg = t.zeros((2, 3, 3, 3))
        >>> seg[:,:,1,1] = 1
        >>> e = t.Tensor([1, 0, 0]).unsqueeze(0).repeat(2, 1)
        >>> t = transform_from_seg(seg, e)
        >>> t["quaternion"]
        tensor([[-0.0000, 0.7070, 0.0000, 0.7075],
                [-0.0000, 0.7070, 0.0000, 0.7075]], dtype=torch.float16)
    """
    assert seg.dim() == 4
    I = affine_identity(dim=seg.dim() - 1, batch_size=seg.shape[0]).to(seg.device)

    # Note that the result of `affine_grid` is somewhat counterintuitive.
    # The first dimension is the batch, the last dimension is the spatial coordinate,
    # and the middle dimensions are the image axes. Though the image axes are ordered
    # as [Caudal -> Cranial] x [Posterior -> Anterior] x [Right -> Left], the ordering
    # of the coordinates in the final vector are reversed. More concretely, the following
    # will all yield constant images:
    #
    # `points_all[0,0,:,:,2]`
    # `points_all[0,:,0,:,1]`
    # `points_all[0,:,:,0,0]`
    #
    # Of course, this ording must be accounted for when interpreting the resulting
    # centroid and rotation.

    points_all = t.nn.functional.affine_grid(
        I, [*seg.unsqueeze(1).shape], align_corners=False
    ).to(seg.device)
    points_all = points_all.flatten(start_dim=1, end_dim=-2)
    weights = seg.flatten(start_dim=1).unsqueeze(-1)

    # https://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis
    centroid = (points_all * weights / weights.sum(axis=1).unsqueeze(-1)).sum(1)
    points_all = points_all - centroid.unsqueeze(1)
    points_weighted = points_all * weights**0.5

    return_dict = normalizing_quaternion_from_points(points_weighted, e)
    return_dict["centroid"] = centroid

    return return_dict
