import os
from argparse import ArgumentParser

from ucca import visualization, layer0
from ucca.convert import split2sentences
from ucca.ioutil import get_passages, get_passages_with_progress_bar, external_write_mode


def print_text(args, text, suffix):
    if args.out_dir:
        with open(os.path.join(args.out_dir, suffix), "w") as f:
            print(text, file=f)
    else:
        with external_write_mode():
            print(text)


def main(args):
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        if not args.tikz:
            import matplotlib
            matplotlib.use('Agg')
    to_stdout = (args.tikz or args.standoff) and not args.out_dir
    t = args.passages
    t = get_passages(t) if to_stdout else get_passages_with_progress_bar(t, desc="Visualizing")
    if args.sentences:
        t = (sentence for passage in t for sentence in split2sentences(passage))
    for passage in t:
        if args.tikz:
            print_text(args, visualization.tikz(passage), passage.ID + ".tikz.txt")
        elif args.standoff:
            print_text(args, visualization.standoff(passage), passage.ID + ".ann")
        else:
            import matplotlib.pyplot as plt
            width = len(passage.layer(layer0.LAYER_ID).all) * 19 / 27
            plt.figure(passage.ID, figsize=(width, width * 10 / 19))
            visualization.draw(passage, node_ids=args.node_ids)
            if args.out_dir:
                plt.savefig(os.path.join(args.out_dir, passage.ID + "." + args.format))
                plt.close()
            else:
                plt.show()


if __name__ == "__main__":
    argparser = ArgumentParser(description="Visualize the given passages as graphs.")
    argparser.add_argument("passages", nargs="+", help="UCCA passages, given as xml/pickle file names")
    group = argparser.add_mutually_exclusive_group()
    group.add_argument("-t", "--tikz", action="store_true", help="print tikz code rather than showing plots")
    group.add_argument("-s", "--standoff", action="store_true", help="print standoff code rather than showing plots")
    argparser.add_argument("-o", "--out-dir", help="directory to save figures in (otherwise displayed immediately)")
    argparser.add_argument("-i", "--node-ids", action="store_true", help="print tikz code rather than showing plots")
    argparser.add_argument("-f", "--format", choices=("png", "svg"), default="png", help="image format")
    argparser.add_argument("--sentences", help="split to sentences to avoid huge plots")
    main(argparser.parse_args())
