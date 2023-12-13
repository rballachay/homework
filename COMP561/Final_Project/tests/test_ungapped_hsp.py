import numpy as np
from ungapped_extension import get_ungapped_hsps


def test_get_ungapped_hsps():
    genome = "TTTTTTTTTTTTTTTTTTTTAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTT"
    probs = np.ones(len(genome))
    query = "GTTTTTTTTTTTTTTAAAAAAAAAAATTTTTTTTTTTTTTG"

    results = get_ungapped_hsps(
        genome, {"AAAAAAAAAAA": [(15, 20), (15, 65)]}, probs, query, 0
    )
    print(results, len(query))
    results = list(results.values())[0][1]
    print(genome[results[1] : results[1] + results[2]])
    print(query[results[0] : results[0] + results[2]])
