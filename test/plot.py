import numpy as np
import matplotlib.pyplot as plt


def plot_2d(img, **kwargs) -> None:
  nz, nx = img.shape
  # Make figure
  fig = plt.figure(figsize=(kwargs.get('wbox', 10), kwargs.get('hbox', 5)))
  ax = fig.gca()
  imin, imax = kwargs.get('imin', np.min(img)), kwargs.get('imax', np.max(img))
  pclip = kwargs.get('pclip', 1.0)
  xmin = kwargs.get('o1', 0.0)
  xmax = kwargs.get('o1', 0.0) + nx * kwargs.get('d1', 1.0)
  zmin = kwargs.get('o2', 0.0)
  zmax = kwargs.get('o2', 0.0) + nz * kwargs.get('d2', 1.0)
  im = ax.imshow(
      img,
      cmap='gray',
      vmin=pclip * imin,
      vmax=pclip * imax,
      interpolation=kwargs.get('interp', 'bilinear'),
      extent=[xmin, xmax, zmax, zmin],
      aspect=kwargs.get('aspect', 1.0),
  )
  ax.set_xlabel(kwargs.get('label1', ' '), fontsize=kwargs.get('labelsize', 15))
  ax.set_ylabel(kwargs.get('label2', ' '), fontsize=kwargs.get('labelsize', 15))
  ax.set_title(kwargs.get('title', ' '), fontsize=kwargs.get('labelsize', 15))
  ax.tick_params(labelsize=kwargs.get('labelsize', 15))
  cbar_ax = fig.add_axes([
      kwargs.get('barx', 0.91),
      kwargs.get('barz', 0.15),
      kwargs.get('wbar', 0.02),
      kwargs.get('hbar', 0.70)
  ])
  cbar = fig.colorbar(im, cbar_ax)
  cbar.ax.tick_params(labelsize=kwargs.get('labelsize', 15))
  # Show the plot
  figname = kwargs.get('figname', None)
  if kwargs.get('show', True):
    plt.show()
  if figname is not None:
    plt.savefig(figname, dpi=150, transparent=True, bbox_inches='tight')


def plot_imgpoff(oimg, dx, dz, zoff, xloc, oh, dh, show=False, **kwargs):
  # Get image dimensions
  nh = oimg.shape[1]
  nz = oimg.shape[2]
  nx = oimg.shape[4]
  fig, ax = plt.subplots(
      1,
      2,
      figsize=(kwargs.get('wbox', 15), kwargs.get('hbox', 8)),
      gridspec_kw={'width_ratios': [12, 1]},
  )
  # Plot the image
  im = ax[0].imshow(
      oimg[0, zoff, :, 0, :],
      extent=[0.0, nx * dx, nz * dz, 0.0],
      interpolation=kwargs.get('interp', 'sinc'),
      cmap=kwargs.get('cmap', 'gray'),
  )
  # Plot a line at the specified image point
  lz = np.linspace(0.0, nz * dz, nz)
  lx = np.zeros(nz) + xloc * dx
  ax[0].plot(lx, lz, color='k', linewidth=2)
  ax[0].set_xlabel('X (km)', fontsize=kwargs.get('labelsize', 14))
  ax[0].set_ylabel('Z (km)', fontsize=kwargs.get('labelsize', 14))
  ax[0].tick_params(labelsize=kwargs.get('labelsize', 14))
  # Plot the extended axis
  ax[1].imshow(
      oimg[0, :, :, 0, xloc].T,
      extent=[oh, kwargs.get('hmax', oh + nh * dh), nz * dz, 0.0],
      interpolation=kwargs.get('interp', 'sinc'),
      cmap=kwargs.get('cmap', 'gray'),
      aspect=1.0,
  )
  ax[1].set_xlabel('Offset (km)', fontsize=kwargs.get('labelsize', 14))
  ax[1].set_ylabel(' ', fontsize=kwargs.get('labelsize', 14))
  ax[1].tick_params(labelsize=kwargs.get('labelsize', 14))
  ax[1].set_yticks([])
  cbar_ax = fig.add_axes([
      kwargs.get('barx', 0.91),
      kwargs.get('barz', 0.32),
      kwargs.get('wbar', 0.02),
      kwargs.get('hbar', 0.35)
  ])
  cbar = fig.colorbar(im, cbar_ax)
  cbar.ax.tick_params(labelsize=kwargs.get('labelsize', 15))
  plt.subplots_adjust(wspace=0.05)
  if show:
    plt.show()
  figname = kwargs.get('figname', None)
  if figname is not None:
    plt.savefig(figname, dpi=150, bbox_inches='tight')


def plot_3d(
    data,
    os=[0.0, 0.0, 0.0],
    ds=[1.0, 1.0, 1.0],
    show=True,
    figname=None,
    **kwargs,
):
  # Transpose if requested
  if not kwargs.get('transp', False):
    data = np.expand_dims(data, axis=0)
    data = np.transpose(data, (0, 1, 3, 2))
  else:
    data = (np.expand_dims(data, axis=-1)).T
    data = np.transpose(data, (0, 1, 3, 2))
  # Get the shape of the cube
  ns = np.flip(data.shape)
  # Make the coordinates for the cross hairs
  ds = np.append(np.flip(ds), 1.0)
  os = np.append(np.flip(os), 0.0)
  x1 = np.linspace(os[0], os[0] + ds[0] * (ns[0]), ns[0])
  x2 = np.linspace(os[1], os[1] + ds[1] * (ns[1]), ns[1])
  x3 = np.linspace(os[2], os[2] + ds[2] * (ns[2]), ns[2])

  # Compute plotting min and max
  vmin = kwargs.get('vmin', None)
  vmax = kwargs.get('vmax', None)
  if (vmin is None or vmax is None):
    vmin = np.min(data) * kwargs.get('pclip', 1.0)
    vmax = np.max(data) * kwargs.get('pclip', 1.0)

  loc1 = kwargs.get('loc1', int(ns[0] / 2 * ds[0] + os[0]))
  i1 = int((loc1 - os[0]) / ds[0])
  loc2 = kwargs.get('loc2', int(ns[1] / 2 * ds[1] + os[1]))
  i2 = int((loc2 - os[1]) / ds[1])
  loc3 = kwargs.get('loc3', int(ns[2] / 2 * ds[2] + os[2]))
  i3 = int((loc3 - os[2]) / ds[2])
  ax1 = None
  ax2 = None
  ax3 = None
  ax4 = None
  curr_pos = 0

  # Axis labels
  label1 = kwargs.get('label1', ' ')
  label2 = kwargs.get('label2', ' ')
  label3 = kwargs.get('label3', ' ')

  width1 = kwargs.get('width1', 4.0)
  width2 = kwargs.get('width2', 4.0)
  width3 = kwargs.get('width3', 4.0)
  widths = [width1, width3]
  heights = [width3, width2]
  gs_kw = dict(width_ratios=widths, height_ratios=heights)
  fig, ax = plt.subplots(
      2,
      2,
      figsize=(width1 + width3, width2 + width3),
      gridspec_kw=gs_kw,
  )
  plt.subplots_adjust(wspace=0, hspace=0)

  title = kwargs.get('title', ' ')
  ax[0, 1].text(
      0.5,
      0.5,
      title[curr_pos],
      horizontalalignment='center',
      verticalalignment='center',
      fontsize=50,
  )

  # xz plane
  ax[1, 0].imshow(
      data[curr_pos, :, i2, :],
      interpolation=kwargs.get('interp', 'none'),
      aspect='auto',
      extent=[os[0], os[0] + (ns[0]) * ds[0], os[2] + ds[2] * (ns[2]), os[2]],
      vmin=vmin,
      vmax=vmax,
      cmap=kwargs.get('cmap', 'gray'),
  )
  ax[1, 0].tick_params(labelsize=kwargs.get('ticksize', 14))
  ax[1, 0].plot(loc1 * np.ones((ns[2],)), x3, c='k')
  ax[1, 0].plot(x1, loc3 * np.ones((ns[0],)), c='k')
  ax[1, 0].set_xlabel(label1, fontsize=kwargs.get('labelsize', 14))
  ax[1, 0].set_ylabel(label3, fontsize=kwargs.get('labelsize', 14))

  # yz plane
  im = ax[1, 1].imshow(
      data[curr_pos, :, :, i1],
      interpolation=kwargs.get('interp', 'none'),
      aspect='auto',
      extent=[os[1], os[1] + (ns[1]) * ds[1], os[2] + (ns[2]) * ds[2], os[2]],
      vmin=vmin,
      vmax=vmax,
      cmap=kwargs.get('cmap', 'gray'),
  )
  ax[1, 1].tick_params(labelsize=kwargs.get('ticksize', 14))
  ax[1, 1].plot(loc2 * np.ones((ns[2],)), x3, c='k')
  ax[1, 1].plot(x2, loc3 * np.ones((ns[1],)), c='k')
  ax[1, 1].get_yaxis().set_visible(False)
  ax[1, 1].set_xlabel(label2, fontsize=kwargs.get('labelsize', 14))
  ax1 = ax[1, 1].twinx()
  ax1.set_ylim(ax[1, 1].get_ylim())
  ax1.set_yticks([loc3])
  ax1.set_yticklabels(['%.2f' % (loc3)], rotation='vertical', va='center')
  ax1.tick_params(labelsize=kwargs.get('ticksize', 14))
  ax2 = ax[1, 1].twiny()
  ax2.set_xlim(ax[1, 1].get_xlim())
  ax2.set_xticks([loc2])
  ax2.set_xticklabels(['%.2f' % (loc2)])
  ax2.tick_params(labelsize=kwargs.get('ticksize', 14))

  # xy plane
  ax[0, 0].imshow(
      np.flip(data[curr_pos, i3, :, :], 0),
      interpolation=kwargs.get('interp', 'none'),
      aspect='auto',
      extent=[os[0], os[0] + (ns[0]) * ds[0], os[1], os[1] + (ns[1]) * ds[1]],
      vmin=vmin,
      vmax=vmax,
      cmap=kwargs.get('cmap', 'gray'),
  )
  ax[0, 0].tick_params(labelsize=kwargs.get('ticksize', 14))
  ax[0, 0].plot(loc1 * np.ones((ns[1],)), x2, c='k')
  ax[0, 0].plot(x1, loc2 * np.ones((ns[0],)), c='k')
  ax[0, 0].set_ylabel(label2, fontsize=kwargs.get('labelsize', 14))
  ax[0, 0].get_xaxis().set_visible(False)
  ax3 = ax[0, 0].twinx()
  ax3.set_ylim(ax[0, 0].get_ylim())
  ax3.set_yticks([loc2])
  ax3.set_yticklabels(['%.2f' % (loc2)], rotation='vertical', va='center')
  ax3.tick_params(labelsize=kwargs.get('ticksize', 14))
  ax4 = ax[0, 0].twiny()
  ax4.set_xlim(ax[0, 0].get_xlim())
  ax4.set_xticks([loc1])
  ax4.set_xticklabels(['%.2f' % (loc1)])
  ax4.tick_params(labelsize=kwargs.get('ticksize', 14))

  # Color bar
  if (kwargs.get('cbar', False)):
    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([
        kwargs.get('barx', 0.91),
        kwargs.get('barz', 0.11),
        kwargs.get('wbar', 0.02),
        kwargs.get('hbar', 0.78)
    ])
    cbar = fig.colorbar(im, cbar_ax, format='%.2f')
    cbar.ax.tick_params(labelsize=kwargs.get('ticksize', 14))
    cbar.set_label(kwargs.get('barlabel', ''),
                   fontsize=kwargs.get("barlabelsize", 13))
    cbar.draw_all()

  ax[0, 1].axis('off')
  if (figname is not None):
    plt.savefig(figname, bbox_inches='tight', transparent=True, dpi=150)
    plt.close()

  if (show):
    plt.show()
