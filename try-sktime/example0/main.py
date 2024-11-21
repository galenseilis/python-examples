"""https://www.sktime.net/en/stable/examples/01c_forecasting_hierarchical_global.html"""

import pandas as pd
# WARN: Accessing protected attribute from external module.
from sktime.utils._testing.hierarchical import _make_hierarchical

from sktime.forecasting.arima import ARIMA

def main() -> None:
    y: pd.DataFrame = _make_hierarchical(hierarchy_levels=(10,3), max_timepoints=1000)

    print(y)

    forecaster = ARIMA()

    y_pred: pd.DataFrame = forecaster.fit(y, fh=[1, 10]).predict()
    print(y_pred)


if __name__ == "__main__":
    main()
