import pandas as pd
import numpy as np


def parse_initial_pop(df: pd.DataFrame) -> np.ndarray:
    ind_cols = list(filter(lambda x: x.startswith("IND"), df.columns))
    population = np.stack(
        [df[i].str.split("|", expand=True).values for i in ind_cols], axis=-1
    )
    return population.astype(np.int)
