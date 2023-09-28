import numpy as np
import argparse


def needleman_wunsch(x, y, match=1, mismatch=1, gap_slip=1, gap_noslip=2):
    nx = len(x)
    ny = len(y)
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:, 0] = np.linspace(0, -nx * gap_noslip, nx + 1)
    F[0, :] = np.linspace(0, -ny * gap_noslip, ny + 1)
    # Pointers to trace through an optimal aligment.
    P = np.zeros((nx + 1, ny + 1))
    P[:, 0] = 3
    P[0, :] = 4
    # Temporary scores.
    t = np.zeros(3)
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i, j] + match
            else:
                t[0] = F[i, j] - mismatch

            # check for equality of the last
            # two elements
            if y[j] == y[j - 1]:
                gap_x = gap_slip
            else:
                gap_x = gap_noslip

            if x[i] == x[i - 1]:
                gap_y = gap_slip
            else:
                gap_y = gap_noslip

            t[1] = F[i, j + 1] - gap_y

            t[2] = F[i + 1, j] - gap_x
            tmax = np.max(t)
            F[i + 1, j + 1] = tmax

            # there are instances where there are multiple maximum
            # scores. this means that there are multiple possible
            # trajectories. this is reflected in the first line
            # of the if statement down below
            if t[0] == tmax:
                P[i + 1, j + 1] += 2
            if t[1] == tmax:
                P[i + 1, j + 1] += 3
            if t[2] == tmax:
                P[i + 1, j + 1] += 4
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    score = 0
    while i > 0 or j > 0:
        if P[i, j] in [2, 9]:
            rx.append(x[i - 1])
            ry.append(y[j - 1])
            i -= 1
            j -= 1
        elif P[i, j] in [3, 5, 7]:
            rx.append(x[i - 1])
            ry.append("-")
            i -= 1
        elif P[i, j] in [4, 6]:
            rx.append("-")
            ry.append(y[j - 1])
            j -= 1

    # Reverse the strings.
    rx = "".join(rx)[::-1]
    ry = "".join(ry)[::-1]
    print(F)
    print(P)
    return "\n".join([rx, ry]), F[-1, -1]


if __name__ == "__main__":
    from Bio import SeqIO

    parser = argparse.ArgumentParser()
    parser.add_argument("-A", help="sequence A", type=str)
    parser.add_argument("-B", help="sequence B", type=str)
    parser.add_argument("-fasta", help="sequence B", type=str)
    parser.add_argument(
        "-match",
        help="score for match, signless",
        type=lambda x: abs(int(x)),
        default=1,
    )
    parser.add_argument(
        "-mismatch",
        help="score for mismatch, signless",
        type=lambda x: abs(int(x)),
        default=1,
    )
    parser.add_argument(
        "-gap_slip", help="score for gap, slip", type=lambda x: abs(int(x)), default=1
    )
    parser.add_argument(
        "-gap_noslip",
        help="score for gap, non-slip",
        type=lambda x: abs(int(x)),
        default=1,
    )
    args = parser.parse_args()

    if args.fasta:
        fasta_sequences = SeqIO.parse(open(args.fasta), "fasta")
        fasta_sequences = [str(i.seq) for i in list(fasta_sequences)]
        assert len(fasta_sequences) == 2

        results, score = needleman_wunsch(
            fasta_sequences[0],
            fasta_sequences[1],
            args.match,
            args.mismatch,
            args.gap_slip,
            args.gap_noslip,
        )
    elif args.A and args.B:
        results, score = needleman_wunsch(
            args.A, args.B, args.match, args.mismatch, args.gap_slip, args.gap_noslip
        )

    print(results)
    print(f"Final score = {score}")
