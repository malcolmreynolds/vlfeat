import numpy as np
import matplotlib.pyplot as plt


def vl_plotframe(frames, color='#00ff00', linewidth=2, eps=1e-9):
    """VL_PLOTFRAME  Plot feature frame
    VL_PLOTFRAME(FRAME) plots the frames FRAME.  Frames are attributed
    image regions (as, for example, extracted by a feature detector). A
    frame is a vector of D=2,3,..,6 real numbers, depending on its
    class. VL_PLOTFRAME() supports the following classes:

     * POINTS
        + FRAME(1:2)   coordinates

     * CIRCLES
        + FRAME(1:2)   center
        + FRAME(3)  radius

     * ORIENTED CIRCLES
        + FRAME(1:2)   center
        + FRAME(3)  radius
        + FRAME(4)  orientation

     * ELLIPSES
        + FRAME(1:2)   center
        + FRAME(3:5)   S11, S12, S22 of x' inv(S) x = 1.

     * ORIENTED ELLIPSES
        + FRAME(1:2)   center
        + FRAME(3:6)   A(:) of ELLIPSE = A UNIT_CIRCLE

    H=VL_PLOTFRAME(...) returns the handle of the graphical object
    representing the frames.

    VL_PLOTFRAME(FRAMES) where FRAMES is a matrix whose column are FRAME
    vectors plots all frames simultaneously. Using this call is much
    faster than calling VL_PLOTFRAME() for each frame.

    VL_PLOTFRAME(FRAMES,...) passes any extra argument to the underlying
    plot function. The first optional argument can be a line
    specification string such as the one used by plt.plot().

    See also:: VL_HELP().

    AUTORIGHTS
    Copyright 2007 (c) Andrea Vedaldi and Brian Fulkerson

    This file is part of VLFeat, available in the terms of the GNU
    General Public License version 2."""

    # number of vertices drawn for each frame
    num_vert = 40

    # --------------------------------------------------------------------
    #                                        Handle various frame classes
    # --------------------------------------------------------------------

    # if just a vector, make sure it is column
    if min(frames.shape) == 1:
        frames = frames[:]

    [D, K] = frames.shape
    zero_dimensional = D == 2

    # just points?
    if zero_dimensional:
        h = plt.plot(frames[0, :], frames[1, :], '.', color=color)
        return

    # reduce all other cases to ellipses/oriented ellipses
    frames = frame2oell(frames, eps=eps)
    do_arrows = (D == 4 or D == 6)

    # Draw
    K = frames.shape[1]
    thr = np.linspace(0, 2 * np.pi, num_vert)

    # allx and ally are nan separated lists of the vertices describing the
    # boundary of the frames
    allx = np.nan * np.ones(([num_vert * K + (K - 1), 1]))
    ally = np.nan * np.ones(([num_vert * K + (K - 1), 1]))

    if do_arrows:
        # allxf and allyf are nan separated lists of the vertices of the
        allxf = np.nan * np.ones([3 * K])
        allyf = np.nan * np.ones([3 * K])

    # vertices around a unit circle
    Xp = np.array([np.cos(thr), np.sin(thr)])

    for k in range(K):
        # frame center
        xc = frames[0, k]
        yc = frames[1, k]

        # frame matrix
        A = frames[2:6, k].reshape([2, 2])

        # vertices along the boundary
        X = np.dot(A, Xp)
        X[0, :] = X[0, :] + xc
        X[1, :] = X[1, :] + yc

        # store
        allx[k * (num_vert + 1) + np.arange(0, num_vert), 0] = X[0, :]
        ally[k * (num_vert + 1) + np.arange(0, num_vert), 0] = X[1, :]

        if do_arrows:
            allxf[k * 3 + np.arange(0, 2)] = xc + [0, A[0, 0]]
            allyf[k * 3 + np.arange(0, 2)] = yc + [0, A[1, 0]]

    if do_arrows:
        for k in range(K):
            h = plt.plot(allx[k * (num_vert + 1) + np.arange(0, num_vert), 0],
                         ally[k * (num_vert + 1) + np.arange(0, num_vert), 0],
                         color=color, linewidth=linewidth)
            h = plt.plot(allxf[k * 3 + np.arange(0, 2)],
                         allyf[k * 3 + np.arange(0, 2)],
                         color=color, linewidth=linewidth)
    else:
        for k in range(K):
            plt.plot(allx[k * (num_vert + 1) + np.arange(0, num_vert), 0],
                     ally[k * (num_vert + 1) + np.arange(0, num_vert), 0],
                     color=color, linewidth=linewidth)


def frame2oell(frames, eps=1e-9):
    """FRAMES2OELL  Convert generic frame to oriented ellipse
      EFRAMES = FRAME2OELL(FRAMES) converts the frames FRAMES to
      oriented ellipses EFRAMES. This is useful because many tasks are
      almost equivalent for all kind of regions and are immediately
      reduced to the most general case."""

    # Determine the kind of frames
    D, K = frames.shape

    if D == 2:
        kind = 'point'
    elif D == 3:
        kind = 'disk'
    elif D == 4:
        kind = 'odisk'
    elif D == 5:
        kind = 'ellipse'
    elif D == 6:
        kind = 'oellipse'
    else:
        raise ValueError('FRAMES format is unknown: %d x %d' % (D, K))

    eframes = np.zeros([6, K])

    # Do converison
    if kind == 'point':
        eframes[0:2, :] = frames[0:2, :]
    elif kind == 'disk':
        eframes[0:2, :] = frames[0:2, :]
        eframes[2, :] = frames[2, :]
        eframes[5, :] = frames[4, :]
    elif kind == 'odisk':
        r = frames[2, :]
        c = r * np.cos(frames[3, :])
        s = r * np.sin(frames[3, :])

        eframes[1, :] = frames[1, :]
        eframes[0, :] = frames[0, :]
        # eframes[2:6, :] = [c, s, - s, c]
        eframes[2:6, :] = [c, -s, s, c]  # not sure why
    elif kind == 'ellipse':
        # sz = find(1e6 * abs(eframes(3,:)) < abs(eframes(4,:)+eframes(5,:))
        eframes[0:2, :] = frames[0:2, :]
        eframes[2, :] = np.sqrt(frames[2, :])
        eframes[3, :] = frames[3, :] / (eframes[2, :] + eps)
        eframes[4, :] = np.zeros([1, K])
        eframes[5, :] = np.sqrt(frames[4, :] -
                                (frames[3, :] ** 2) / (frames[2, :] + eps))
    elif kind == 'oellipse':
        eframes = frames

    return eframes


def vl_plotsiftdescriptor(descs, frames, color='g',
                          num_spatial_bins=4, num_orient_bins=8,
                          magnif=3, linewidth=1, maxv=None):
    """VL_PLOTSIFTDESCRIPTOR(D) plots the SIFT descriptors D, stored as
    columns of the matrix D. D has the same format used by VL_SIFT().

    VL_PLOTSIFTDESCRIPTOR(D,F) plots the SIFT descriptors warped to
    the SIFT frames F, specified as columns of the matrix F. F has the
    same format used by VL_SIFT().

    H=VL_PLOTSIFTDESCRIPTOR(...) returns the handle H to the line drawing
    representing the descriptors.

    REMARK. By default, the function assumes descriptors with 4x4
    spatial bins and 8 orientation bins (Lowe's default.)

    The function supports the following options

    NumSpatialBins:: [4]
       Number of spatial bins in each spatial direction.

    NumOrientBins:: [8]
       Number of orientation bis.

    Magnif:: [3]
    Magnification factor.

    See also: VL_SIFT(), VL_PLOTFRAME(), VL_HELP().

    Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
    All rights reserved.

    This file is part of the VLFeat library and is made available under
    the terms of the BSD license (see the COPYING file)."""

    desc_size, num_descs = descs.shape
    if desc_size != (num_spatial_bins ** 2) * num_orient_bins:
        raise ValueError('number of rows of descs')

    if frames is None:
        frames = np.tile(np.array([[0, 0, 1, 0]]).T, (1, num_descs))

    frame_type, num_frames = frames.shape

    if num_frames != num_descs:
        raise ValueError("number of descriptors != number of frames")

    if frame_type == 2:
        frames = np.vstack([frames, 10 * np.ones(num_frames),
                            np.zeros(num_frames)])
    elif frame_type == 3:
        frames = np.vstack([frames, np.zeros(num_frames)])
    elif frame_type != 4:
        raise ValueError('frames.shape[0] must be in [2, 3, 4]')

    x_all, y_all = np.array([]), np.array([])

    descs = descs.astype(np.float)

    for k in range(num_descs):
        sbp = magnif * frames[2, k]
        theta = frames[3, k]
        c, s = np.cos(theta), np.sin(theta)

        xs, ys = render_descr(descs[:, k], num_spatial_bins, num_orient_bins,
                              maxv=maxv)
        new_x = sbp * (c*xs - s*ys) + frames[0, k]
        new_y = sbp * (s*xs + c*ys) + frames[1, k]

        x_all = np.hstack([x_all, new_x])
        y_all = np.hstack([y_all, new_y])
        # plt.plot(new_x, new_y)

    plt.plot(x_all, y_all, c=color, linewidth=linewidth)


def render_descr(desc, num_spatial_bins, num_orient_bins, maxv=None, eps=1e-6):
    """render an individual SIFT descriptor"""

    hnsb = num_spatial_bins / 2
    x, y = np.meshgrid(np.r_[-hnsb:hnsb + 1], np.r_[-hnsb:(hnsb + 1)])

    if maxv is not None:
        desc *= (0.4 / maxv)
    else:
        desc *= (0.4 / (desc.max() + eps))

    # desc_size = desc.size
    # desc = np.zeros(desc_size)

    # num_to_draw = 8
    # desc[:num_to_draw] = np.r_[:num_to_draw]
    # import ipdb; ipdb.set_trace()

    # Bin centres - offset by a half from coords except for last row / column
    xc = (x[:-1, :-1] + 0.5).flatten()
    yc = (y[:-1, :-1] + 0.5).flatten()

    # import ipdb; ipdb.set_trace()

    # Each bin centre contains a start with num_orient_bins tips
    xc = np.tile(xc.reshape(num_spatial_bins ** 2, 1), (1, num_orient_bins))
    yc = np.tile(yc.reshape(num_spatial_bins ** 2, 1), (1, num_orient_bins))
    # yc = np.tile(yc, (num_orient_bins, 1))

    # Make the stars
    th = np.linspace(0, 2 * np.pi, num_orient_bins + 1)[:-1]
    xd = np.tile(np.cos(th), (1, num_spatial_bins ** 2))
    yd = np.tile(np.sin(th), (1, num_spatial_bins ** 2))

    xd = xd * desc.flatten()
    yd = yd * desc.flatten()

    # Rearrange in sequential order
    nans = np.nan * np.ones((1, num_spatial_bins ** 2 * num_orient_bins))
    start_x = xc.flatten()
    start_y = yc.flatten()

    end_x = start_x + xd.flatten()
    end_y = start_y + yd.flatten()

    xstars = np.vstack([start_x, end_x, nans])
    ystars = np.vstack([start_y, end_y, nans])

    # Horizontal grid lines
    nans = np.nan * np.ones((1, num_spatial_bins + 1))
    xh = np.vstack([x[:, 0].flatten(), x[:, -1].flatten(), nans])
    yh = np.vstack([y[:, 0].flatten(), y[:, -1].flatten(), nans])

    # Vertical lines
    xv = np.vstack([x[0, :].flatten(), x[-1, :].flatten(), nans])
    yv = np.vstack([y[0, :].flatten(), y[-1, :].flatten(), nans])

    final_x = np.hstack([xstars, xh, xv])
    final_y = np.hstack([ystars, yh, yv])

    return final_x.T.flatten(), final_y.T.flatten()



    # return xstars.T.flatten(), ystars.T.flatten()


