"""
Assembles image using reference and a bank of stars
"""
from PIL import Image
import glob
import os
import random
import numpy as np
import copy
import a107
import argparse
from starmelib import *
import tabulate
from IPython import embed


__all__ = ["STARMEDATADIR", "Session", "list_banks_"]


STARMEDATADIR = "/home/j/.starme"


class Point:
    @property
    def x0(self):
        return self.x-self.width//2

    @property
    def y0(self):
        return self.y-self.height//2

    def __init__(self, session, x, y, index):
        self.session = session
        self.x = x
        self.y = y
        self.index = index
        self.img = session.bank[index]
        self.aimg = np.asarray(self.img)
        self.width = self.aimg.shape[1]
        self.height = self.aimg.shape[0]
        self.mass = np.sum(self.aimg[:,:,3] > 0)
        self.vel = np.array([0.,0])
        self.force = np.array([0.,0])
        self.acc = np.array([0.,0])


class Session(a107.ConsoleCommands):
    def __init__(self, args):
        super().__init__()

        self.datadir = args.datadir
        self.bankname = ""

        # REFERENCE IMAGE
        self.img = self.aimg = self.w = self.h = self.spawn_count = self.ind = None

        # BANK
        self.bank = []
        self.masses = []
        self.weights = []
        self.ladder = None
        self.nb = None
        self._load_bank(args.bankname)

        if args.input:
            self._load(args.input)

    def _load_bank(self, bankname):
        if self.img:
            if not a107.yesno("Loading another bank will reset the project. Continue", 0):
                return

        self.bank = []
        self.masses = []
        self.weights = []

        fulldatadir = os.path.join(self.datadir, bankname)
        gg = glob.glob(os.path.join(fulldatadir, "*.png"))
        for g in gg:
            img = Image.open(g, 'r')

            if False:
                # changes alpha
                G = list(img.split())
                G[-1] = G[-1].point(lambda x: x*.5)
                img = Image.merge("RGBA", G)

            self.bank.append(img)
            mass = np.sum(np.asarray(img)[:, :, :3])
            self.masses.append(mass)
            self.weights.append(1/mass)
        self.ladder = np.cumsum(self.weights)
        self.nb = len(self.bank)
        self.bankname = bankname
        print(f"Loaded {self.nb} image{'s' if self.nb != 1 else ''} from '{fulldatadir}")

    def random_index(self):
        x = random.random()*self.ladder[-1]
        index = a107.BSearchCeil(self.ladder, x)
        return index

    def spawn(self, amount=1):
        """Spawns stars, but only if it is better with them than without them."""
        amount = int(amount)
        for _ in range(amount):
            x = random.randint(0, self.w-1)
            y = random.randint(0, self.h-1)
            index = self.random_index()
            p = Point(self, x, y, index)
            p.on = False
            mark_off = self._evaluate_point(p)
            p.on = True
            mark_on = self._evaluate_point(p)
            len0 = len(self.ind)
            if mark_on < mark_off:
                self.ind.append(p)
            print(f"spawn #{self.spawn_count} {'accepted' if len(self.ind) > len0 else 'rejected'}")
            self.spawn_count += 1

    def improve(self, amount=1):
        amount = int(amount)
        print("####################", amount)
        for _i in range(amount):
            print("aaaaaaaaaaaaaaaaaa", _i)
            for i in range(len(self.ind)):
                p = self.ind[i]
                error0 = self._evaluate_point(p)
                q = copy.copy(p)
                self._mutate_point(q)
                error1 = self._evaluate_point(q)
                if error1 is None:
                    continue
                if error1 < error0:
                    self.ind[i] = q

    def show(self, flag_imgref=False):
        """Builds the resulting image and shows it."""
        flag_imgref = a107.to_bool(flag_imgref)
        img = self._make_image(flag_imgref)
        img.show()

    def show_img(self):
        """Shows the reference image."""
        self.img.show()

    def draw_bank(self, ratio=1.3, flag_save=False, backgroundcolor=(0, 0, 0, 255)):
        """
        Builds a grid-like image containing all the bank images with each cell in the grid containing a "star".

        Args:
            ratio: ratio between columns and rows in image
            flag_save: whether or not to save image to disk

        """
        img = self._make_draw_bank(ratio, backgroundcolor)
        if flag_save:
            filename = a107.new_filename(f"grid-{self.bankname}", "png")
            img.save(filename)
            print(f"Saved file '{filename}'")
        img.show()

    def save(self, flag_imgref=False):
        flag_imgref = a107.to_bool(flag_imgref)
        img = self._make_image(flag_imgref)
        filename = a107.new_filename("nowstars0", "png", False)
        img.save(filename)
        return filename

    def list_banks(self):
        list_banks_(self.datadir)

    def load_bank(self, bankname):
        self._load_bank(bankname)

    def load(self, inputfilename):
        """Loads image"""
        self._load(inputfilename)

    def _load(self, inputfilename):
        if self.img:
            if not a107.yesno("Loading another image will reset the project. Continue", 0):
                return

        self.img = Image.open(inputfilename, 'r')
        self.aimg = np.asarray(self.img)
        self.w, self.h = self.img.size
        self.spawn_count = 0
        self.ind = self._get_individual()

    def _get_individual(self):
        return []

    def _mycrop(self, box):
        """
        Crops array and recalculates the box.

        The box will be recalculated if its coordinates are outside the image boundaries.

        Returns:
            cropped_array, box_offsets
        """
        if box[0] >= self.w or box[2] < 0 or box[1] >= self.h or box[3] < 0:
            return None, None
        imgwidth, imgheight = self.img.size
        x0 = max(0, box[0])
        y0 = max(0, box[1])
        x1 = min(imgwidth, box[2])
        y1 = min(imgheight, box[3])
        off_x0 = max(0, x0 - box[0])
        off_y0 = max(0, y0 - box[1])
        off_x1 = min(0, x1 - box[2])
        off_y1 = min(0, y1 - box[3])
        return self.aimg[y0:y1, x0:x1], [off_x0, off_y0, off_x1, off_y1]

    def _evaluate_point(self, p=None):
        """
        Evaluates "point". The lower the better

        Sum of squared distances. If the reference is white and the "point" is white, the
        evaluation/"mark" is 0.
        """
        error = None

        acropped, offsets = self._mycrop([p.x0, p.y0, p.x0 + p.width, p.y0 + p.height])

        if acropped is None:
            return None

        ai = p.aimg
        if any(offsets):
            ai = p.aimg[offsets[1]:p.height+offsets[3], offsets[0]:p.width+offsets[2]]

        w, h = ai.shape[:2]
        if w == 0 or h == 0:
            return None

        mask = ai[:,:,-1] > 0  # == np.max(ai[:,:,-1])  # pixels of opacity > 0
        nmask = np.sum(mask)

        if p.on:
            aii = ai[mask, :3]
        else:
            aii = 0

        ac = np.array(acropped[mask,:3], dtype=float)
        try:
            error = np.sum(np.abs(ac-aii))/(nmask)
        except:
            raise

        return error

    def _get_paste_data(self, p):
        """Calculates and returns information used to paste image onto another image.

        Args:
            p: Point instance

        Returns:
            img, pos_x, pos_y, width, heigth
        """

        img = self.bank[p.index]
        if p.angle != 0:
            img = img.rotate(p.angle)
        width, height = img.size
        pos_x = p.x - width // 2
        pos_y = p.y - height // 2

        return img, pos_x, pos_y, width, height

    def _mutate_point(self, p):
        HALFSHIT = 5
        if random.random() < 0.2:
            p.x += random.randint(-HALFSHIT, HALFSHIT)
        if random.random() < 0.2:
            p.y += random.randint(-HALFSHIT, HALFSHIT)
        # if random.random() < 0.1:
        #     p.on = not p.on

    def _make_image(self, flag_imgref):
        background = self.img.copy() if flag_imgref else Image.new("RGBA", (self.w, self.h), (0, 0, 0, 255))

        for p in self.ind:
            if p.on:
                background.paste(p.img, (p.x0, p.y0), p.img.split()[-1])

        return background

    def _make_draw_bank(self, ratio, backgroundcolor):
        # calculates size of cell in grid
        maxw, maxh = 0, 0
        for star in self.bank:
            w, h = star.size
            maxw = max(w, maxw)
            maxh = max(h, maxh)
        dim = max(maxw, maxh)

        # calculates number of cells
        n = self.nb
        nr = int(np.floor(np.sqrt(n/ratio)))
        nc = int(np.ceil(n/nr))

        MARGIN = 3  # outer box
        SPACING = 2  # spacing between stars in grid

        width = nc*dim+2*MARGIN+(nc-1)*SPACING
        height = nr*dim+2*MARGIN+(nr-1)*SPACING

        img = Image.new("RGBA", (width, height), backgroundcolor)

        x = y = MARGIN
        i = 0
        for star in self.bank:
            w, h = star.size
            x0 = int(x+(dim-w)/2)
            y0 = int(y+(dim-h)/2)
            img.paste(star, (x0, y0), star.split()[-1])

            x += SPACING+dim
            i += 1
            if i == nc:
                y += SPACING+dim
                x = MARGIN
                i = 0

        return img


def list_banks_(datadir):
    """Lists all image banks in data directory."""
    dd = glob.glob(os.path.join(datadir, "*"))
    rows = []
    header = ["Bank name", "Directory", "Number of PNG files"]
    for d in dd:
        if os.path.isdir(d):
            n = len(glob.glob(os.path.join(d, "*.png")))
            rows.append([os.path.split(d)[1], d, n])
    print(tabulate.tabulate(rows, header))
