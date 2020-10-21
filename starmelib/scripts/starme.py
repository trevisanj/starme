"""
Assembles image using reference and a bank of stars
"""
import a107
import argparse
from starmelib import *
from IPython import embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=a107.SmartFormatter)
    parser.add_argument("-d", "--datadir", default=STARMEDATADIR,
                        help="Data directory, the place containing the image bank and where to save the commands history")
    parser.add_argument("-b", "--bankname", default="sky-0012")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input filename")
    parser.add_argument("-l", "--list-banks", action="store_true", help="Lists all available image banks")

    args = parser.parse_args()

    if args.list_banks:
        list_banks_(args.datadir)
    else:
        S = Session(args)
        a107.publish_in_pythonconsole(S, globals())
        embed()
        # k = a107.Console("starme", S, args.datadir)
        # k.run()
