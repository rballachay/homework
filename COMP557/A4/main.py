import numpy
import argparse
import matplotlib.pyplot as plt
import timeit

# Ported from C++ by Melissa Katz
# Adapted from code by Loïc Nassif and Paul Kry

parse = argparse.ArgumentParser()
parse.add_argument("--infile", type=str, help="Name of json file that will define the scene")
parse.add_argument("--outfile", type=str, default="out.png", help="Name of png that will contain the render")
parse.add_argument("--numba", action='store_true', help="Use numba just-in-time compilation")
args = parse.parse_args()

if __name__ == "__main__":

    start = timeit.default_timer()

    if args.numba:
        import provided_numba.scene_parser as scene_parser
        full_scene = scene_parser.load_scene(args.infile)
    else:
        import provided.scene_parser as scene_parser
        full_scene = scene_parser.load_scene(args.infile)

    image = full_scene.render()
    image = numpy.rot90(image, k=1, axes=(0, 1))
    plt.axis("off")
    plt.imshow(image)
    #plt.show()
    plt.savefig(args.outfile, bbox_inches='tight')

    stop = timeit.default_timer()
    print(f'Program run time: {(stop-start):.2f}')  
