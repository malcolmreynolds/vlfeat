#!/usr/bin/env ipython --pylab -i
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import vlfeat
from vlfeat.plotop.vl_plot import plot_frame, plot_sift_descriptor
from numpy.random import shuffle

import skimage.data as skd


if __name__ == '__main__':
    # VL_DEMO_SIFT_BASIC  Demo: SIFT: basic functionality
    # --------------------------------------------------------------------
    #                     Load a figure and convert the to required format
    # --------------------------------------------------------------------

    I = vlfeat.vl_rgb2gray(skd.lena())
    I = numpy.array(I, 'f', order='F')  # 'F' = column-major order!

    plt.figure()
    plt.imshow(I, cmap=cm.Greys_r)

    # Run SIFT
    f, d = vlfeat.vl_sift(I)

    # plot a randomly chosen set of 50 frames
    num_sifts_to_plot = 50
    sel = numpy.arange(f.shape[1])
    shuffle(sel)
    # sel = np.array([0], dtype=np.int)
    plot_frame(f[:, sel[:num_sifts_to_plot]], color='k', linewidth=3)
    plot_frame(f[:, sel[:num_sifts_to_plot]], color='y')
    plot_sift_descriptor(d[:, sel[:num_sifts_to_plot]],
                         f[:, sel[:num_sifts_to_plot]])
    plt.title('%d randomly chosen SIFT frames' % num_sifts_to_plot)

#   h3 = plot_sift_descriptor(d(:,sel),f(:,sel)) ;
#   set(h3,'color','k','linewidth',2) ;
#   h4 = plot_sift_descriptor(d(:,sel),f(:,sel)) ;
#   set(h4,'color','g','linewidth',1) ;
#   h1   = plot_frame(f(:,sel)) ; set(h1,'color','k','linewidth',3) ;
#   h2   = plot_frame(f(:,sel)) ; set(h2,'color','y','linewidth',2) ;
#   vl_demo_print('sift_basic_3') ;

    # --------------------------------------------------------------------
    #                                                      Custom keypoint
    # --------------------------------------------------------------------
    plt.figure()
    plt.imshow(I[:200, :200]) 


    fc = numpy.array([99, 99, 10, -numpy.pi/8], 'float64')
    fc = numpy.array(numpy.atleast_2d(fc).transpose(), order='F')
    f, d = vlfeat.vl_sift(I, frames=fc, verbose=False)


#   h3   = plot_sift_descriptor(d,f) ;  set(h3,'color','k','linewidth',3) ;
#   h4   = plot_sift_descriptor(d,f) ;  set(h4,'color','g','linewidth',2) ;
    plot_frame(f, color='k', linewidth=4)
    plot_frame(f, color='y', linewidth=2)
    
    print 'sift_basic_4'

    # --------------------------------------------------------------------
    #                                   Custom keypoints with orientations
    # --------------------------------------------------------------------
    
    fc = numpy.array([99, 99, 10, 0], 'float64')
    fc = numpy.array(numpy.atleast_2d(fc).transpose(), order='F')
    f, d = vlfeat.vl_sift(I, frames=fc, orientations=True) ;
    
#   h3   = plot_sift_descriptor(d,f) ;  set(h3,'color','k', 'linewidth',3) ;
#   h4   = plot_sift_descriptor(d,f) ;  set(h4,'color','g', 'linewidth',2) ;
    plot_frame(f, color='k', linewidth=4)
    plot_frame(f, color='y', linewidth=2)
    
    print 'sift_basic_5'

    pylab.show()
    
    
    