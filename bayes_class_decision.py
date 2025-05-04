import pandas as pd
import numpy as np


def save_parameters_to_csv(parameter_file_path, mean, std):
    """
    Save mean and standard‑deviation values to a CSV file.

    Parameters
    ----------
    parameter_file_path : str
        Destination path for the parameter file.
    mean : float
        Estimated mean of the feature.
    std : float
        Estimated standard deviation of the feature.
    """
    parameters_df = pd.DataFrame(
        {
            "Parameter": ["mean", "std"],
            "Values": [mean, std],
        }
    )
    parameters_df.to_csv(parameter_file_path, index=False)


def estimate_gaussian_parameters(file_path,
                                 Parameter_file_path,
                                 feature_name="Values"):
    """
    Estimate Gaussian parameters (mean & std) for a given feature column
    in *file_path* and save them to *Parameter_file_path*.

    Parameters
    ----------
    file_path : str
        CSV file that contains the sample data.
    Parameter_file_path : str
        CSV file where the estimated parameters will be stored.
    feature_name : str, optional
        Column name of the feature to analyse (default: "Values").
    """
    data = pd.read_csv(file_path)
    feature_values = data[feature_name].values

    mean = np.mean(feature_values)
    std = np.std(feature_values)

    save_parameters_to_csv(Parameter_file_path, mean, std)


def get_parameters(parameter_file_path):
    """
    Read mean and std values from *parameter_file_path*.

    Returns
    -------
    tuple(float, float)
        (mean, std)
    """
    data = pd.read_csv(parameter_file_path)
    mean = data[data["Parameter"] == "mean"]["Values"].values[0]
    std = data[data["Parameter"] == "std"]["Values"].values[0]
    return mean, std


def predict_group(value,
                  mean_clean, std_clean,
                  mean_dirty, std_dirty):
    """
    Classify *value* as 'Clean' or 'G' (dirty) using
    univariate Gaussian likelihoods.
    """
    p_clean = (1 / (np.sqrt(2 * np.pi) * std_clean)) * np.exp(
        -0.5 * ((value - mean_clean) / std_clean) ** 2
    )
    p_dirty = (1 / (np.sqrt(2 * np.pi) * std_dirty)) * np.exp(
        -0.5 * ((value - mean_dirty) / std_dirty) ** 2
    )
    return "Clean" if p_clean > p_dirty else "G"


if __name__ == "__main__":
    # Example usage
    new_value = 0.67

    clean_file = "data_100_150.csv"
    dirty_file = "data_100_150_dirty.csv"
    clean_parameters_file = "clean_parameters.csv"
    dirty_parameters_file = "dirty_parameters.csv"

    # Estimate parameters and write them to disk
    estimate_gaussian_parameters(clean_file, clean_parameters_file)
    estimate_gaussian_parameters(dirty_file, dirty_parameters_file)

    # Retrieve parameters for the classifier
    mean_clean, std_clean = get_parameters(clean_parameters_file)
    mean_dirty, std_dirty = get_parameters(dirty_parameters_file)

    # Predict the group for *new_value*
    predicted_group = predict_group(
        new_value,
        mean_clean, std_clean,
        mean_dirty, std_dirty
    )

    print(f"Number {new_value} belongs to group: {predicted_group}")
