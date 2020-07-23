#!/usr/bin/env python
"""
Decomposes a night sky photograph into its individual stars to make a "star bank".

Creates a directory and saves each star as a separate .png file.
"""
from PIL import Image
import a107
import numpy as np
import os
import csv
import argparse


AA = 1
COLORTHRESHOLD = 50
YI = 32

class Point:
    def __init__(self, x, y, s, filename):
        self.x = x
        self.y = y
        self.s = s
        self.filename = filename
        self.mass = None
        self.vel = [0,0]
        self.acc = [0,0]


class Session:
    def __init__(self, aimg):
        self.aimg = aimg
        self.restart()

    def restart(self):
        """Restarts from aimg."""
        self.hei, self.wid, _ = self.aimg.shape
        self.shit = np.sum(self.aimg, 2) > COLORTHRESHOLD
        self.been = np.zeros((self.hei, self.wid), dtype=bool)


    def find_sth(self, x, y):
        """Returns box containing something.

        **note** right ends are non-pythonic
        """
        ret = [999999, 999999, -2, -2]
        npo = [0]

        def _find_sth(xi, yi):
            xl = max(0, xi-AA)
            xr = min(self.wid-1, xi+AA)
            yl = max(0, yi-AA)
            yr = min(self.hei-1, yi+AA)
            found = False

            ff = []
            for yii in range(yl, yr+1):
                for xii in range(xl, xr+1):
                    if not self.been[yii, xii]:
                        self.been[yii, xii] = True
                        if self.shit[yii, xii]:
                            found = True
                            if ret[0] > xii: ret[0] = xii
                            if ret[2] < xii: ret[2] = xii
                            if ret[1] > yii: ret[1] = yii
                            if ret[3] < yii: ret[3] = yii
                            npo[0] += 1
                            ff.append((xii, yii))


            for f in ff:
                _find_sth(f[0], f[1])

        _find_sth(x, y)

        return ret


def main(args):
    import matplotlib.pyplot as plt
    # gg = glob.glob(os.path.join(DATADIR, "*"))
    # filename = gg[0]
    filename = args.input
    img = Image.open(filename, "r")
    if False:
        img.show()
    aimg = np.asarray(img)
    if False:
        aimgd = np.diff(aimg, 1, 1)
    if False:
        plt.imshow(aimgd)
        plt.show()
    if False:
        imgdc = Image.fromarray(aimgd)
        imgdc.show()
    o = Session(aimg)
    hei, wid, _ = aimg.shape
    ostia = []
    yi = AA
    while yi < hei:
        xi = AA
        ly = []
        while xi < wid:
            x0, y0, x1, y1 = o.find_sth(xi, yi)
            if x1 - x0 > 0 and y1 - y0 > 0 and x1 - x0 + y1 - y0 > 1:
                s = np.s_[y0:y1 + 1, x0:x1 + 1]
                if False:
                    anave = aimg[s]
                    plt.imshow(anave)
                    plt.show()
                if False:
                    anave = o.shit[s]
                    plt.matshow(anave)
                    plt.show()

                ostia.append(Point((x1 + x0) // 2, (y1 + y0) // 2, s, ""))

                xi = x1 + 1
                ly.append(y1)

            else:
                xi += 2 * AA + 1

        if ly:
            yi = max(ly) + 1
        else:
            yi += 2 * AA + 1

        # if len(ostia) > 10:
        #     break
    if False:
        _, axarr = plt.subplots(4, 5)

        k = 0
        for j in range(4):
            for i in range(5):
                a = axarr[j][i]
                if k < len(ostia):
                    if False:
                        a.matshow(o.shit[ostia[k].s])
                    else:
                        a.imshow(o.aimg[ostia[k].s])
                k += 1
                a.axis("off")

        plt.show()
    print(f"Found {len(ostia)} objects")
    savedir = "."
    dirname = a107.new_filename(os.path.join(savedir, "sky"), flag_minimal=False)
    os.mkdir(dirname)
    for i, p in enumerate(ostia):
        outputfilename = a107.new_filename(os.path.join(dirname, "ostia"), "png", flag_minimal=False)
        p.filename = outputfilename
        print(f"Saving {outputfilename}...")

        M = o.aimg[p.s]
        h, w, _ = M.shape
        MA = np.zeros((h, w, 4), dtype="uint8")
        MA[:, :, :3] = M
        MA[:, :, 3] = o.shit[p.s] * 255

        Image.fromarray(MA).save(outputfilename)
        print("...done")
    csvfilename = os.path.join(dirname, "metadata.csv")
    wr = csv.writer(open(csvfilename, "w"))
    wr.writerows([(os.path.split(p.filename)[1], p.x, p.y) for p in ostia])
    print(f"Saved {csvfilename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=a107.SmartFormatter)
    parser.add_argument("input", type=str, help="Filename to extract the stars from")


    args = parser.parse_args()

    main(args)
