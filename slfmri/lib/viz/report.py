import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


class BaseReport(object):
    def __init__(self):
        pass

    def set_axes(self, n_row, n_col, title=None, *args, **kwargs):
        fig, axes = plt.subplots(n_row, n_col, *args, **kwargs)
        if title is not None:
            plt.suptitle(title, fontsize=self.font_size['title'])
        return fig, axes

    def plot_ts(self, ax, ts_data, *args, **kwargs):
        if isinstance(ts_data, pd.DataFrame):
            n_data = ts_data.shape[0]
        elif isinstance(ts_data, pd.Series):
            n_data = 1
        else:
            raise Exception # unexpected data type

        if n_data == 1:
            ax.plot(ts_data, *args, **kwargs)
        else:
            for i, col in enumerate(ts_data.columns):
                col_kwargs = dict()
                for key in kwargs.keys():
                    if isinstance(kwargs[key], dict):
                        col_kwargs[key] = kwargs[key][col]
                    elif isinstance(kwargs[key], list):
                        col_kwargs[key] = kwargs[key][i]
                    else:
                        if hasattr(kwargs[key], '__iter__'):
                            raise Exception
                        else:
                            col_kwargs[key] = kwargs[key]
                if 'label' in kwargs.keys():
                    ax.plot(ts_data[col], *args, **col_kwargs)
                else:
                    ax.plot(ts_data[col], *args, label=col, **col_kwargs)

    @property
    def layout(self):
        return dict(us_letter=(8.5, 11))

    @property
    def preset(self):
        return dict(dpi=150, linewidth=2)

    @property
    def font_size(self):
        return dict(title=15, label=13, tick=12, legend=10)


class QC_report(BaseReport):
    """
    The class to plot QC report
    """
    def __init__(self, qc_object=None):
        super(QC_report, self).__init__()
        self._fd = None
        self._dvars = None
        self._vwi = None
        self._std = None
        self._tsnr = None
        self._mparam = None

        if qc_object is not None:
            self.set_qc(qc_object)
        else:
            pass

    @property
    def mparam(self):
        return self._mparam

    @property
    def fd(self):
        return self._fd

    @property
    def dvars(self):
        return self._dvars

    @property
    def vwi(self):
        return self._vwi

    @property
    def std(self):
        return self._std

    @property
    def tsnr(self):
        return self._tsnr

    @property
    def mean_fd(self):
        return

    @property
    def rms(self):
        return

    def set_qc(self, qc_object):
        self._fd = qc_object.FD.loc[:, ::-1]
        self._dvars = qc_object.DVARS.loc[:, ::-1]
        self._vwi = qc_object.VWI
        self._std = qc_object.STD
        self._tsnr = qc_object.tSNR
        self._mparam = qc_object.mparam.loc[:, ::-1]

    def plot_MotionAndBold(self, subj_code):
        from ..utils import get_rgb_tuple
        title = u'{}\nMean FD: {} µm\nRMS movements: {} µm'.format(subj_code, self.mean_fd, self.rms)

        fig, axes = self.set_axes(5, 1, sharex=True, title=title,
                                  figsize=self.layout['us_letter'], dpi=self.preset['dpi'],
                                  gridspec_kw={'height_ratios': [1, 1, 1, 1, 2], 'width_ratios': [1]})

        # Motion parameters
        mp_linestyle = ['-'] * 3 + [':'] * 3
        mp_color_map = {0: get_rgb_tuple(188, 182, 247),
                        1: get_rgb_tuple(200, 188, 139),
                        2: get_rgb_tuple(205, 137, 134)}
        mp_color_map = mp_color_map.values() * 2
        self.plot_ts(axes[0], self.mparam * 1000, linewidth=self.preset['linewidth'], linestyle=mp_linestyle,
                     color=mp_color_map)

        # Framewise displacement
        fd_color_map = {'FD': get_rgb_tuple(188, 47, 40),
                        'ATD': get_rgb_tuple(137, 33, 28),
                        'ARD': get_rgb_tuple(151, 60, 56)}
        fd_labels = {'FD': 'Framewise\nDisplacement (FD)',
                     'ATD': 'Absolute translational\ndisplacement (ATD)',
                     'ARD': 'Absolute rotational\ndisplacement (ARD)'}
        fd_linestyles = ['-', '--', ':']
        fd_alpha = [1 - i * 0.2 for i in range(3)]
        self.plot_ts(axes[1], self.fd * 1000, linewidth=self.preset['linewidth'], linestyle=fd_linestyles,
                     color=fd_color_map, label=fd_labels, alpha=fd_alpha)

        # DVARS
        self.plot_ts(axes[2], self.dvars['DVgs'], label='DVgs', color='blue', linewidth=self.preset['linewidth'])

        # SDgs and Global signal
        self.plot_ts(axes[3], self.dvars['SDgs'], label='SDgs',
                     color='green', linewidth=self.preset['linewidth'])
        self.plot_ts(axes[3], self.dvars['GS'], label='Global signal',
                     color='black', linewidth=self.preset['linewidth'])

        # Voxel intensity
        vwi_plot = axes[4].imshow(self.vwi, aspect='auto', cmap='gray', origin='lower',
                                  vmin=-20, vmax=20)
        position = fig.add_axes([0.75, 0.11, 0.02, 0.20])
        cb = plt.colorbar(vwi_plot, cax=position, ticks=[-20, 0, 20])
        cb.ax.tick_params(labelsize=self.font_size['legend'])

        # Legend and label control
        labels = [u'µm'] * 2 + ['BOLD'] * 2 + ['Voxels']
        label_coord = -0.1, 0.5
        legend_loc = [(1.02, 0.7), (1.02, 1.0), (1.02, 0.7), (1.02, 0.7), (1.02, 0.9)]
        legend_col = [2, 1, 1, 1, 1]

        for i in range(5):
            axes[i].legend(loc=2, bbox_to_anchor=legend_loc[i], ncol=legend_col[i])
            axes[i].get_yaxis().set_label_coords(*label_coord)
            axes[i].yaxis.set_major_locator(plt.MaxNLocator(3))
            axes[i].set_ylabel(labels[i], fontsize=self.font_size['label'])
        axes[4].set_xlabel('Volumes #', fontsize=self.font_size['label'])
        axes[4].get_xaxis().set_label_coords(0.5, -0.2)
        axes[4].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.rc('xtick', labelsize=self.font_size['tick'])
        plt.rc('ytick', labelsize=self.font_size['tick'])
        plt.rc('legend', fontsize=self.font_size['legend'], borderaxespad=0, frameon=False)

        # Set data range
        axes[0].set_xlim([0, self.mparam.shape[0]])
        axes[0].set_ylim([-50, 50])
        if any(self.fd.max() > 100):
            pass
        else:
            axes[1].set_ylim([0, 100])
        if self.dvars['DVgs'].max() > 50:
            pass
        else:
            axes[2].set_ylim([0, 50])
        if self.dvars['SDgs'].max() > 40 or self.dvars['SDgs'].min() < -20:
            pass
        else:
            axes[3].set_ylim([-20, 40])

        # Layout control
        fig.subplots_adjust(top=0.9, left=0.15, right=0.70, bottom=0.1)

    def save_pdf(self, subj_code, output_fname):
        with PdfPages(output_fname) as pdf:
            plt.figure()
            self.plot_MotionAndBold(subj_code)
            pdf.savefig()
            plt.close()