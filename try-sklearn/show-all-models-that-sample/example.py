import pandas as pd
import inspect
from sklearn.utils import all_estimators


def get_models_with_methods():
    models_with_methods = []

    # Get all estimators from sklearn
    estimators = all_estimators()

    for name, EstimatorClass in estimators:
        # Check if EstimatorClass is a class
        if inspect.isclass(EstimatorClass):
            has_sample = "sample" in dir(EstimatorClass)
            has_sample_y = "sample_y" in dir(EstimatorClass)
            has_predict_proba = "predict_proba" in dir(EstimatorClass)
            if has_sample or has_sample_y or has_predict_proba:
                models_with_methods.append(
                    (name, has_sample, has_sample_y, has_predict_proba)
                )

    return models_with_methods


if __name__ == "__main__":
    models_with_methods = get_models_with_methods()
    df = pd.DataFrame(
        models_with_methods,
        columns=["Model Class", "`sample`", "`sample_y`", "`predict_proba`"],
    )
    print(df.to_markdown(index=False))
