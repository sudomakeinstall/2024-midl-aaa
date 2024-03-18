# System
import pathlib as pl
import datetime as dt
import itertools

# Third Party
import itk
import tomlkit as tk
import numpy as np
import matplotlib as mpl

mpl.use("QtAgg")
import matplotlib.pyplot as plt
import torch as t

# Internal
from . import geometry


class DiscreteColormapGenerator:
    def __init__(self, labels):
        self.labels = labels
        self.labels.sort(key=lambda x: x.integer)

    def get_colormap(self):
        return mpl.colors.ListedColormap([l.color for l in self.labels])

    def get_normalizer(self):
        bins = [l.integer for l in self.labels]
        bins = [self.labels[0].integer - 1] + bins + [self.labels[-1].integer + 1]
        bins = np.array(bins[0:-1] + bins[1:]).reshape(2, -1).mean(0)
        return mpl.colors.BoundaryNorm(bins, len(self.labels), clip=True)

    def get_legend_patches(self):
        return [
            mpl.patches.Patch(color=l.color, label=l.label)
            for l in self.labels
            if l.legend
        ]


class WindowLevel:
    def __init__(self, name, window, level):
        self.name = name
        self.window = window
        self.level = level

    def lower(self):
        return self.level - self.window / 2

    def upper(self):
        return self.level + self.window / 2


class WindowLevelPresets:
    def __init__(self, index="7"):
        self.index = index
        self.presets = {
            "1": WindowLevel("abdomen", 400, 40),
            "2": WindowLevel("lung", 1500, -700),
            "3": WindowLevel("liver", 100, 110),
            "4": WindowLevel("bone", 1500, 500),
            "5": WindowLevel("brain", 85, 42),
            "6": WindowLevel("stroke", 36, 28),
            "7": WindowLevel("vascular", 800, 200),
            "8": WindowLevel("subdural", 160, 60),
        }

    def preset(self):
        return self.presets[self.index]

    def lower(self):
        return self.preset().lower()

    def upper(self):
        return self.preset().upper()


class Viewer3D(object):
    def __init__(
        self,
        im,
        mk=None,
        labels=None,
        slices=[0],
        scroll="clamp",
        background="black",
        legend_loc="lower center",
        legend_visible=True,
    ):
        self.im = im
        self.mk = mk
        if self.mk is not None:
            self.cmpr = DiscreteColormapGenerator(labels)
            self.legend_loc = legend_loc
            self.legend_visible = legend_visible

        self.slices = slices
        self.scroll = scroll
        if self.scroll not in {"clamp", "wrap"}:
            self.scroll = "clamp"
        self.wlp = WindowLevelPresets()
        self.wlp.presets["9"] = WindowLevel(
            "minmax",
            self.im.max() - self.im.min(),
            (self.im.max() - self.im.min()) / 2 + self.im.min(),
        )

        self.fig, self.axes = plt.subplots(1, len(self.slices), squeeze=False)
        self.axes = self.axes[0]
        for ax in self.axes:
            ax.axis("off")
        self.fig.patch.set_facecolor(background)
        self.ix = [self.im.shape[s] // 2 for s in self.slices]

        self.axes_im = [
            ax.imshow(np.take(self.im, i, axis=s), cmap="gray", origin="lower")
            for ax, i, s in zip(self.axes, self.ix, self.slices)
        ]
        if self.mk is not None:
            self.axes_mk = [
                ax.imshow(
                    np.take(self.mk, i, axis=s),
                    cmap=self.cmpr.get_colormap(),
                    norm=self.cmpr.get_normalizer(),
                    interpolation="nearest",
                    origin="lower",
                )
                for ax, i, s in zip(self.axes, self.ix, self.slices)
            ]
            if self.legend_visible:
                self.fig.legend(
                    handles=self.cmpr.get_legend_patches(), loc=self.legend_loc, ncol=3
                )

        mpl_disconnect_callbacks(self.fig)

        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)
        self.fig.canvas.mpl_connect("key_press_event", self.onpress)
        plt.tight_layout()
        plt.show()

    def get_active_axis(self, event):
        active = np.flatnonzero([event.inaxes is ax for ax in self.axes])
        if len(active) == 0:
            return
        return active[0]

    def onscroll(self, event):
        active = self.get_active_axis(event)
        if active is None:
            return

        if event.button == "up":
            self.ix[active] += 1
        else:
            self.ix[active] -= 1
        self.update(active)

    def onpress(self, event):
        if event.key == "h":
            print(
                """
  h : Print available keybindings.
  s : Save plot.
  q : Localize all plots to the point indicated by the cursor.
  [1-9] : Select window/level preset.
  {j, down} : Scroll down.
  {k, up} : Scroll up.
"""
            )
        if event.key == "s":
            timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            path = f"{timestamp}.png"
            self.fig.savefig(path, dpi=600)
            print(f"Saved plot to: {path}")
            return

        active = self.get_active_axis(event)
        if active is None:
            return

        if event.key == "q":
            indices = [int(event.xdata), int(event.ydata)]
            indices.insert(self.slices[active], self.ix[active])
            indices = np.roll(indices, self.slices[active])
            indices = [indices[0], indices[2], indices[1]]
            self.ix = [indices[s] for s in self.slices]
        elif event.key in self.wlp.presets.keys():
            self.wlp.index = event.key
            self.axes_im[active].set_clim(self.wlp.lower(), self.wlp.upper())
        elif event.key in {"j", "down"}:
            self.ix[active] -= 1
        elif event.key in {"k", "up"}:
            self.ix[active] += 1

        self.update_all()

    def update_all(self):
        for i in range(len(self.slices)):
            self.update(i)

    def update(self, active):
        if self.scroll == "clamp":
            self.ix[active] = max(
                min(self.ix[active], self.im.shape[self.slices[active]] - 1), 0
            )
        elif self.scroll == "wrap":
            self.ix[active] %= self.im.shape[self.slices[active]]
        else:
            print("Option not recognized.")

        self.axes_im[active].set_data(
            np.take(self.im, self.ix[active], axis=self.slices[active])
        )
        self.axes_im[active].axes.figure.canvas.draw()

        if self.mk is not None:
            self.axes_mk[active].set_data(
                np.take(self.mk, self.ix[active], axis=self.slices[active])
            )
            self.axes_mk[active].axes.figure.canvas.draw()


def mpl_disconnect_callbacks(figure):
    cids = []
    for _, event_dict in figure.canvas.callbacks.callbacks.items():
        for k in event_dict.keys():
            cids += [k]
    for c in cids:
        figure.canvas.callbacks.disconnect(c)
