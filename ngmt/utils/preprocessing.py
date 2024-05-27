# Import libraries
import importlib.resources as pkg_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal
import scipy.io
import scipy.integrate
import scipy.ndimage
import pywt
from ngmt.utils import quaternion
from ngmt.config import cfg_colors


# use the importlib.resources package to access the FIR_2_3Hz_40.mat file
with pkg_resources.path(
    "ngmt.utils", "FIR_2_3Hz_40.mat"
) as mat_filter_coefficients_file:
    pass


def resample_interpolate(
    input_signal, initial_sampling_frequency: float, target_sampling_frequency: float
):
    """
    Resample and interpolate a signal to a new sampling frequency.

    This function takes a signal `input_signal` sampled at an initial sampling frequency `initial_sampling_frequency`
    and resamples it to a target sampling frequency `target_sampling_frequency` using linear interpolation.

    Args:
        input_signal (array_like): The input signal.
        initial_sampling_frequency (float, optional): The initial sampling frequency of the input signal. Default is 100.
        target_sampling_frequency (float, optional): The target sampling frequency for the output signal. Default is 40.

    Returns:
        resampled_signal (array_like): The resampled and interpolated signal.
    """
    # Error handling for invalid input data
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal should be a NumPy array.")

    if (
        not isinstance(initial_sampling_frequency, (int, float))
        or initial_sampling_frequency <= 0
    ):
        raise ValueError("The initial sampling frequency must be a positive float.")

    if (
        not isinstance(target_sampling_frequency, (int, float))
        or target_sampling_frequency <= 0
    ):
        raise ValueError("The target sampling frequency must be a positive float.")

    # Calculate the length of the input signal.
    recording_time = len(input_signal)

    # Create an array representing the time indices of the input signal.
    x = np.arange(1, recording_time + 1)

    # Create an array representing the time indices of the resampled signal.
    xq = np.arange(
        1, recording_time + 1, initial_sampling_frequency / target_sampling_frequency
    )

    # Create an interpolation function using linear interpolation and apply it to the data.
    interpolator = scipy.interpolate.interp1d(
        x, input_signal, kind="linear", axis=0, fill_value="extrapolate"
    )

    # Resample and interpolate the input signal to the desired target sampling rate.
    resampled_signal = interpolator(xq)

    return resampled_signal


def lowpass_filter(signal, method="savgol", order=None, **kwargs):
    """
    Apply a low-pass filter to the input signal.

    Args:
        signal (numpy.ndarray): The input signal to be filtered.
        method (str): The filter method to use ("savgol", "butter", or "fir").
        order (int): The order of the filter (applicable for "butter" method).
        param (**kwargs): Additional keyword arguments specific to the Savitzky-Golay filter method or other methods.

    Returns:
        filt_signal (numpy.ndarray): The filtered signal.
    """
    # Error handling for invalid input data
    if not isinstance(signal, np.ndarray):
        raise ValueError("Input data must be a numpy.ndarray")

    if not isinstance(method, str):
        raise ValueError("'method' must be a string.")

    method = method.lower()

    # Define default parameters for Savitzky-Golay filter
    default_savgol_params = {
        "window_length": 21,
        "polynomial_order": 7,
    }

    # Define default parameters for FIR filter
    default_fir_params = {
        "fir_file": mat_filter_coefficients_file,
    }

    if method == "savgol":
        # Update default parameters with any provided kwargs
        savgol_params = {**default_savgol_params, **kwargs}
        window_length = savgol_params.get(
            "window_length", default_savgol_params["window_length"]
        )
        polynomial_order = savgol_params.get(
            "polynomial_order", default_savgol_params["polynomial_order"]
        )

        filt_signal = scipy.signal.savgol_filter(
            signal, window_length, polynomial_order
        )
        return filt_signal

    elif method == "butter":
        # Extract parameters specific to butterworth filter
        cutoff_freq_hz = kwargs.get("cutoff_freq_hz", 5.0)
        sampling_rate_hz = kwargs.get("sampling_rate_hz", 200.0)

        if order is None:
            raise ValueError("For Butterworth filter, 'order' must be specified.")

        # Apply butterworth lowpass filter
        b, a = scipy.signal.butter(
            N=order,
            Wn=cutoff_freq_hz / (sampling_rate_hz / 2),
            btype="low",
            analog=False,
            fs=sampling_rate_hz,
        )
        filt_signal = scipy.signal.filtfilt(b, a, signal)
        return filt_signal

    elif method == "fir":
        # Update default parameters with any provided kwargs
        fir_params = {**default_fir_params, **kwargs}
        fir_file = fir_params.get("fir_file", default_fir_params["fir_file"])

        # Load FIR low-pass filter coefficients from the specified MAT file
        lowpass_coefficients = scipy.io.loadmat(fir_file)
        numerator_coefficient = lowpass_coefficients["Num"][0, :]

        # Define the denominator coefficients as [1.0] to perform FIR filtering
        denominator_coefficient = np.array([1.0])

        # Apply the FIR low-pass filter using filtfilt
        filtered_signal = scipy.signal.filtfilt(
            numerator_coefficient, denominator_coefficient, signal
        )

        return filtered_signal

    else:
        raise ValueError("Invalid filter method specified")


def highpass_filter(signal, sampling_frequency=40, method="iir", **kwargs):
    """
    Apply a high-pass filter to the input signal using the specified method.

    Args:
        signal (np.ndarray): The input signal to be filtered.
        sampling_frequency (float): The sampling frequency of the input signal.
        method (str): The filtering method to be used.
        **kwargs: Additional keyword arguments specific to the filtering method.

    Returns:
        np.ndarray: The filtered signal.

    """
    # Error handling for invalid input data
    if (
        not isinstance(signal, np.ndarray)
        or not isinstance(sampling_frequency, (int, float))
        or sampling_frequency <= 0
    ):
        raise ValueError(
            "Invalid input data. The 'signal' must be a NumPy array, and 'sampling_frequency' must be a positive number."
        )

    if not isinstance(method, str):
        raise ValueError("'method' must be a string.")

    method = method.lower()

    if method == "iir":
        filtered_signal = _iir_highpass_filter(signal, sampling_frequency)
    else:
        raise ValueError(f"Unsupported filtering method: {method}")

    return filtered_signal


def _iir_highpass_filter(signal, sampling_frequency=40):
    """
    Apply an IIR high-pass filter to the input signal.

    Args:
        signal (np.ndarray): The input signal to be filtered.
        sampling_frequency (float): The sampling frequency of the input signal.

    Returns:
        np.ndarray: The filtered signal.

    Notes:
        The FIR filter coefficients are loaded from the specified MAT file (`fir_file`).
        The filter is applied using `scipy.signal.filtfilt`, which performs zero-phase
        filtering to avoid phase distortion.
    """
    # Error handling for invalid input data
    if (
        not isinstance(signal, np.ndarray)
        or not isinstance(sampling_frequency, (int, float))
        or sampling_frequency <= 0
    ):
        raise ValueError(
            "Invalid input data. The 'signal' must be a NumPy array, and 'sampling_frequency' must be a positive number."
        )

    if sampling_frequency == 40:
        # The numerator coefficient vector of the high-pass filter.
        numerator_coefficient = np.array([1, -1])

        # The denominator coefficient vector of the high-pass filter.
        denominator_coefficient = np.array([1, -0.9748])

        # Apply the FIR low-pass filter using filtfilt
        filtered_signal = scipy.signal.filtfilt(
            numerator_coefficient,
            denominator_coefficient,
            signal,
            axis=0,
            padtype="odd",
            padlen=3
            * (max(len(numerator_coefficient), len(denominator_coefficient)) - 1),
        )

    # Return the filtered signal

    return filtered_signal


def apply_continuous_wavelet_transform(
    data, scales=10, desired_scale=10, wavelet="gaus2", sampling_frequency=40
):
    """
    Apply continuous wavelet transform to the input signal.

    Args:
        data (numpy.ndarray): Input signal.
        scales (int, optional): Number of scales for the wavelet transform. Default is 10.
        desired_scale (int, optional): Desired scale to use in calculations. Default is 10.
        wavelet (str, optional): Type of wavelet to use. Default is 'gaus2'.
        sampling_frequency (float, optional): Sampling frequency of the signal. Default is 40.

    Returns:
        smoothed_data (numpy.ndarray): Smoothed data after applying multiple Gaussian filters.
    """
    # Error handling for invalid input data
    try:
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy.ndarray")
        if not isinstance(scales, int) or scales <= 0:
            raise ValueError("Scales must be a positive integer")
        if not isinstance(sampling_frequency, (int, float)) or sampling_frequency <= 0:
            raise ValueError("Sampling frequency must be a positive number")

        sampling_period = 1 / sampling_frequency
        coefficients, _ = pywt.cwt(
            data, np.arange(1, scales + 1), wavelet, sampling_period
        )
        wavelet_transform_result = coefficients[desired_scale - 1, :]

        return wavelet_transform_result
    except Exception as e:
        # Handle the exception by printing an error message and returning None.
        print(f"Error in apply_continuous_wavelet_transform: {e}")

        return None


def apply_successive_gaussian_filters(data):
    """
    Apply successive Gaussian filters to the input data.

    Args:
        data (numpy.ndarray): Input data.

    Returns:
        data (numpy.ndarray): Filtered data.
    """
    # Error handling for invalid input data
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")

    if data.size < 1:
        raise ValueError("Input data must not be empty.")

    sigma_params = [2, 2, 3, 2]
    kernel_size_params = [10, 10, 15, 10]
    mode_params = ["reflect", "reflect", "nearest", "reflect"]

    filtered_signal = data

    for sigma, kernel_size, mode in zip(sigma_params, kernel_size_params, mode_params):

        gaussian_radius = (kernel_size - 1) / 2
        filtered_signal = scipy.ndimage.gaussian_filter1d(
            filtered_signal, sigma=sigma, mode=mode, radius=round(gaussian_radius)
        )

    return filtered_signal


def calculate_envelope_activity(
    input_signal, smooth_window=20, threshold_style=1, duration=20
):
    """
    Calculate envelope-based activity detection using the Hilbert transform.

    This function analyzes an input signal `input_signal` to detect periods of activity based on the signal's envelope.
    It calculates the analytical signal using the Hilbert transform, smoothes the envelope, and applies an
    adaptive threshold to identify active regions.

    Parameters:
        input_signal (array_like): The input signal.
        smooth_window (int): Window length for smoothing the envelope (default is 20).
        threshold_style (int): Threshold selection style: 0 for manual, 1 for automatic (default is 1).
        duration (int): Minimum duration of activity to be detected (default is 20).

    Returns:
        tuple (ndarray, ndarray): A tuple containing:
        alarm (ndarray): Vector indicating active parts of the signal.
        env (ndarray): Smoothed envelope of the signal.
    """
    # Error handling for invalid input data
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input signal should be a NumPy array.")

    if not isinstance(smooth_window, (int)) or smooth_window <= 0:
        raise ValueError("The window length must be a positive integer.")

    if not isinstance(threshold_style, (int)) or threshold_style <= 0:
        raise ValueError("The threshold style must be a positive integer.")

    if not isinstance(duration, (int)) or duration <= 0:
        raise ValueError("The duration must be a positive integer.")

    # Calculate the analytical signal and get the envelope
    input_signal = input_signal.flatten()
    # Compute the analytic signal, using the Hilbert transform form scipy.signal.
    analytic = scipy.signal.hilbert(input_signal)
    env = np.abs(analytic)  # Compute the envelope of the analytic signal.

    # Take the moving average of the analytic signal
    env = scipy.signal.convolve(
        env, np.ones(smooth_window) / smooth_window, mode="full"
    )  # Returns the discrete, linear convolution of two one-dimensional sequences.

    env = env - np.mean(env)  # Remove the offset by subtracting the mean of 'env'
    env = env / np.max(env)  # Normalize the 'env' by dividing by its maximum value

    # Threshold the signal
    # if threshold_style == 0:
    #     plt.plot(env)
    #     plt.title("Select a threshold on the graph")
    #     THR_SIG = plt.ginput(1)[0][1]
    #     plt.close()
    # else:
    #     THR_SIG = 4 * np.mean(env)
    THR_SIG = 4 * np.mean(env)

    # Set noise and signal levels
    noise = np.mean(env) / 3  # noise level

    # Signal level: It's used as a reference to distinguish between the background noise and the actual signal activity.
    threshold = np.mean(env)

    # Initialize Buffers
    thres_buf = np.zeros(
        len(env) - duration
    )  # This buffer stores values related to a threshold.
    noise_buf = np.zeros(
        len(env) - duration
    )  # This buffer stores values related to the noise.
    THR_buf = np.zeros(len(env))  # This buffer stores threshold values.
    alarm = np.zeros(len(env))  # This buffer tracks alarm-related information.
    h = 1

    for i in range(len(env) - duration):
        if np.all(env[i : i + duration + 1] > THR_SIG):
            alarm[i] = np.max(
                env
            )  # If the current window of data surpasses the threshold, set an alarm.
            threshold = 0.1 * np.mean(
                env[i : i + duration + 1]
            )  # Set a new threshold based on the mean of the current window.
            h += 1
        else:
            # Update noise
            if np.mean(env[i : i + duration + 1]) < THR_SIG:
                noise = np.mean(
                    env[i : i + duration + 1]
                )  # Update the noise value based on the mean of the current window.
            else:
                if len(noise_buf) > 0:
                    noise = np.mean(
                        noise_buf
                    )  # If available, use the mean of noise buffer to update the noise.
        thres_buf[i] = threshold  # Store the threshold value in the threshold buffer.
        noise_buf[i] = noise  # Store the noise value in the noise buffer.

        # Update threshold
        if h > 1:
            THR_SIG = noise + 0.50 * (
                np.abs(threshold - noise)
            )  # Update the threshold using noise and threshold values.
        THR_buf[i] = (
            THR_SIG  # Store the updated threshold value in the threshold buffer.
        )

    # if plot_results == 1:
    #     plt.figure()
    #     ax = plt.subplot(2, 1, 1)
    #     plt.plot(input_signal)
    #     plt.plot(np.where(alarm != 0, np.max(input_signal), 0), "r", linewidth=2.5)
    #     plt.plot(THR_buf, "--g", linewidth=2.5)
    #     plt.title("Raw Signal and detected Onsets of activity")
    #     plt.legend(
    #         ["Raw Signal", "Detected Activity in Signal", "Adaptive Threshold"],
    #         loc="upper left",
    #     )
    #     plt.grid(True)
    #     plt.axis("tight")

    #     ax2 = plt.subplot(2, 1, 2)
    #     plt.plot(env)
    #     plt.plot(THR_buf, "--g", linewidth=2.5)
    #     plt.plot(thres_buf, "--r", linewidth=2)
    #     plt.plot(noise_buf, "--k", linewidth=2)
    #     plt.title("Smoothed Envelope of the signal (Hilbert Transform)")
    #     plt.legend(
    #         [
    #             "Smoothed Envelope of the signal (Hilbert Transform)",
    #             "Adaptive Threshold",
    #             "Activity level",
    #             "Noise Level",
    #         ]
    #     )
    #     plt.grid(True)
    #     plt.axis("tight")
    #     plt.tight_layout()
    #     plt.show()

    return alarm, env


def find_consecutive_groups(input_signal):
    """
    Find consecutive groups of non-zero values in an input array.

    This function takes an input array `input_signal`, converts it to a column vector, and identifies consecutive groups of
    non-zero values. It returns a 2D array where each row represents a group, with the first column containing
    the start index of the group and the second column containing the end index of the group.

    Parameters:
        input_array (ndarray): The input array.

    Returns:
        ind (ndarray): A 2D array where each row represents a group of consecutive non-zero values.
            The first column contains the start index of the group, and the second column contains the end index.
    """
    # Error handling for invalid input data
    if not isinstance(input_signal, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")

    if input_signal.size < 1:
        raise ValueError("Input data must not be empty.")

    # Find indices of non-zeros elements
    temp = np.where(input_signal)[0]

    # Find where the difference between indices is greater than 1
    idx = np.where(np.diff(temp) > 1)[0]

    # Initialize the output array
    ind = np.zeros((len(idx) + 1, 2), dtype=int)

    # Set the second column
    ind[:, 1] = temp[np.append(idx, -1)]

    # Set the first column
    ind[:, 0] = temp[np.insert(idx + 1, 0, 0)]

    return ind


def find_local_min_max(signal, threshold=None):
    """
    Find Local Minima and Maxima in a Given Signal.

    This function takes an input signal and identifies the indices of local minima and maxima.
    Optionally, a threshold can be provided to filter out minima and maxima that do not exceed the threshold.

    Args:
        signal (numpy.ndarray): The input signal.
        threshold (float or None, optional): Threshold for filtering out minima and maxima below and above this value, respectively.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray): A tuple containing two arrays:
            - minima_indices: Indices of local minima in the signal.
            - maxima_indices: Indices of local maxima in the signal.
    """
    # Error handling for invalid input data
    if not isinstance(signal, np.ndarray):
        raise ValueError("Input signal must be a NumPy array.")

    if signal.size < 1:
        raise ValueError("Input signal must not be empty.")

    # Find positive peaks in the signal
    maxima_indices, _ = scipy.signal.find_peaks(signal)

    # Find negative peaks in the inverted signal
    minima_indices, _ = scipy.signal.find_peaks(-signal)

    if threshold is not None:
        maxima_indices = maxima_indices[signal[maxima_indices] > threshold] + 1
        minima_indices = minima_indices[signal[minima_indices] < -threshold] + 1

    return minima_indices, maxima_indices


def identify_pulse_trains(signal):
    """
    Identify Pulse Trains in a Given Signal.

    This function takes an input signal and detects pulse trains within the signal.
    A pulse train is identified as a sequence of values with small intervals between adjacent values.

    Args:
        signal (numpy.ndarray): The input signal.

    Returns:
        pulse_train (list): A list of dictionaries, each containing information about a detected pulse train.
            Each dictionary has the following keys:

            `start`: The index of the first value in the pulse train.

            `end`: The index of the last value in the pulse train.

            `steps`: The number of steps in the pulse train.
    """
    # Error handling for invalid input data
    if signal.size < 1:
        raise ValueError("Input signal must not be empty.")

    # Initialize an empty list to store detected pulse trains.
    pulse_trains = []

    # Initialize a flag to track whether we are within a pulse train.
    walking_flag = 0

    # Set an initial threshold value for pulse train detection.
    threshold = 3.5 * 40

    # Initialize a counter for the number of detected pulse trains.
    pulse_count = 0

    # Check if the signal has more than 2 elements.
    if len(signal) > 2:
        for i in range(len(signal) - 1):
            # Check if the difference between adjacent values is less than the threshold.
            if signal[i + 1] - signal[i] < threshold:
                if walking_flag == 0:
                    # If not already in a pulse train, start a new one.
                    pulse_trains.append({"start": signal[i], "steps": 1})
                    pulse_count += 1
                    walking_flag = 1
                else:
                    # If already in a pulse train, update the number of steps and threshold.
                    pulse_trains[pulse_count - 1]["steps"] += 1
                    threshold = (
                        1.5 * 40
                        + (signal[i] - pulse_trains[pulse_count - 1]["start"])
                        / pulse_trains[pulse_count - 1]["steps"]
                    )
            else:
                if walking_flag == 1:
                    # If leaving a pulse train, record its end and reset threshold.
                    pulse_trains[pulse_count - 1]["end"] = signal[i - 1]
                    walking_flag = 0
                    threshold = 3.5 * 40

    if walking_flag == 1:
        if signal[-1] - signal[-2] < threshold:
            # If still in a pulse train at the end, record its end and update steps.
            pulse_trains[-1]["end"] = signal[-1]
            pulse_trains[-1]["steps"] += 1
        else:
            # If leaving a pulse train at the end, record its end.
            pulse_trains[-1]["end"] = signal[-2]

    return pulse_trains


def convert_pulse_train_to_array(pulse_train_list):
    """
    Convert a List of Pulse Train Dictionaries to a 2D Array.

    This function takes a list of pulse train dictionaries and converts it into a 2D array.
    Each dictionary is expected to have keys 'start' and 'end', and the function creates an array
    where each row corresponds to a dictionary with the 'start' value in the first column and the
    'end' value in the second column.

    Args:
        pulse_train_list (list): A list of dictionaries containing pulse train information.

    Returns:
        array_representation(numpy.ndarray): A 2D array where each row represents a pulse train with the 'start' value
                                            in the first column and the 'end' value in the second column.
    """
    # Error handling for invalid input data
    if not isinstance(pulse_train_list, list):
        raise ValueError("Input should be a list of pulse train dictionaries.")

    # Check if the list is empty
    if not pulse_train_list:
        raise ValueError("Input list is empty.")

    # Check that each element in the list is a dictionary with the expected keys
    for pulse_train in pulse_train_list:
        if not isinstance(pulse_train, dict):
            raise ValueError("Each element in the list should be a dictionary.")
        if "start" not in pulse_train or "end" not in pulse_train:
            raise ValueError("Each dictionary should contain 'start' and 'end' keys.")

    # Initialize a 2D array with the same number of rows as pulse train dictionaries and 2 columns.
    array_representation = np.zeros((len(pulse_train_list), 2), dtype=np.uint64)

    # Iterate through the list of pulse train dictionaries.
    for i, pulse_train_dict in enumerate(pulse_train_list):
        array_representation[i, 0] = pulse_train_dict["start"]
        array_representation[i, 1] = pulse_train_dict["end"]

    return array_representation


def find_interval_intersection(set_a, set_b):
    """
    Find the Intersection of Two Sets of Intervals.

    Given two sets of intervals, this function computes their intersection and returns a new set
    of intervals representing the overlapping regions.

    Args:
        set_a (numpy.ndarray): The first set of intervals, where each row represents an interval with two values
            indicating the start and end points.
        set_b (numpy.ndarray): The second set of intervals, with the same structure as `set_a`.

    Returns:
        intersection_intervals (numpy.ndarray): A new set of intervals representing the intersection of intervals from `set_a` and `set_b`.
    """
    # Error handling for invalid input data
    if not isinstance(set_a, np.ndarray) or not isinstance(set_b, np.ndarray):
        raise ValueError("Both input sets should be NumPy arrays.")

    # Check if the input sets have the correct structure (two columns)
    if set_a.shape[1] != 2 or set_b.shape[1] != 2:
        raise ValueError(
            "Input sets should have two columns, indicating start and end points."
        )

    # Get the number of intervals in each set.
    num_intervals_a = set_a.shape[0]
    num_intervals_b = set_b.shape[0]

    # Initialize an empty list to store the intersection intervals.
    intersection_intervals = []

    # If either set of intervals is empty, return an empty array.
    if num_intervals_a == 0 or num_intervals_b == 0:
        return np.array(intersection_intervals)

    # Initialize indices and state variables for set_a and set_b traversal.
    index_a = 0
    index_b = 0
    state = 3

    # Traverse both sets of intervals and compute their intersection.
    while index_a < num_intervals_a and index_b < num_intervals_b:
        if state == 1:
            if set_a[index_a, 1] < set_b[index_b, 0]:
                index_a += 1
                state = 3
            elif set_a[index_a, 1] < set_b[index_b, 1]:
                intersection_intervals.append([set_b[index_b, 0], set_a[index_a, 1]])
                index_a += 1
                state = 2
            else:
                intersection_intervals.append(set_b[index_b, :])
                index_b += 1
        elif state == 2:
            if set_b[index_b, 1] < set_a[index_a, 0]:
                index_b += 1
                state = 3
            elif set_b[index_b, 1] < set_a[index_a, 1]:
                intersection_intervals.append([set_a[index_a, 0], set_b[index_b, 1]])
                index_b += 1
                state = 1
            else:
                intersection_intervals.append(set_a[index_a, :])
                index_a += 1
        elif state == 3:
            if set_a[index_a, 0] < set_b[index_b, 0]:
                state = 1
            else:
                state = 2

    return np.array(intersection_intervals)


def organize_and_pack_results(walking_periods, peak_steps):
    """Organize and Pack Walking Results with Associated Peak Steps.

    Given lists of walking periods and peak step indices, this function organizes and packs the results
    into a more structured format. It calculates the number of steps in each walking period, associates
    peak steps with their corresponding walking periods, and extends the duration of walking periods based
    on step time. The function also checks for overlapping walking periods and merges them.

    Args:
        walking_periods (list): List of tuples representing walking periods, where each tuple contains the start and end indices.
        peak_steps (list): List of peak step indices.

    Returns:
        organized_results (list): A list of dictionaries representing organized walking results, each dictionary contains:

                - 'start': Start index of the walking period.

                - 'end': End index of the walking period.

                - 'steps': Number of steps within the walking period.

                - 'mid_swing': List of peak step indices within the walking period.

        all_mid_swing (list): A list of sorted peak step indices across all walking periods.
    """
    # Calculate the number of walking periods.
    num_periods = len(walking_periods)

    # Initialize a list of dictionaries to store organized walking results.
    organized_results = [
        {
            "start": walking_periods[i][0],
            "end": walking_periods[i][1],
            "steps": 0,
            "mid_swing": [],
        }
        for i in range(num_periods)
    ]

    # Initialize a list to store all peak step indices.
    all_mid_swing = []

    # Iterate through each walking period.
    for i in range(num_periods):
        # Find peak steps within the current walking period.
        steps_within_period = [
            p
            for p in peak_steps
            if organized_results[i]["start"] <= p <= organized_results[i]["end"]
        ]

        # Calculate the number of steps within the walking period.
        organized_results[i]["steps"] = len(steps_within_period)

        # Store the peak step indices within the walking period.
        organized_results[i]["mid_swing"] = steps_within_period

        # Add peak step indices to the list of all peak step indices.
        all_mid_swing.extend(steps_within_period)

        # Calculate step time based on detected peak steps
        if len(steps_within_period) > 2:
            step_time = sum(
                [
                    steps_within_period[j + 1] - steps_within_period[j]
                    for j in range(len(steps_within_period) - 1)
                ]
            ) / (len(steps_within_period) - 1)
            organized_results[i]["start"] = int(
                organized_results[i]["start"] - 1.5 * step_time / 2
            )
            organized_results[i]["end"] = int(
                organized_results[i]["end"] + 1.5 * step_time / 2
            )

    # Sort all peak step indices.
    all_mid_swing.sort()

    # Check for overlapping walking periods and merge them
    i = 0
    while i < num_periods - 1:
        if organized_results[i]["end"] >= organized_results[i + 1]["start"]:
            organized_results[i]["end"] = organized_results[i + 1]["end"]
            organized_results[i]["steps"] += organized_results[i + 1]["steps"]
            organized_results[i]["mid_swing"].extend(
                organized_results[i + 1]["mid_swing"]
            )
            organized_results.pop(i + 1)
            num_periods -= 1
        else:
            i += 1

    return organized_results, all_mid_swing


def max_peaks_between_zc(input_signal):
    """
    Find peaks and their locations from the vector input_signal between zero crossings.

    Args:
        input_signal (numpy.ndarray): Input column vector.

    Returns:
        pks (numpy.ndarray): Signed max/min values between zero crossings.
        ipks (numpy.ndarray): Locations of the peaks in the original vector.
    """
    # Flatten the input vector to ensure it's 1D.
    input_signal = input_signal.flatten()

    # Find the locations of zero crossings in the input vector.
    zero_crossings_locations = (
        np.where(np.abs(np.diff(np.sign(input_signal))) == 2)[0] + 1
    )

    # Calculate the number of peaks.
    number_of_peaks = len(zero_crossings_locations) - 1

    def imax(signal_segment):
        return np.argmax(signal_segment)

    # Find the indices of the maximum values within each peak region.
    ipk = np.array(
        [
            imax(
                np.abs(
                    input_signal[
                        zero_crossings_locations[i]:zero_crossings_locations[i + 1]
                    ]
                )
            )
            for i in range(number_of_peaks)
        ]
    )

    # Compute the locations of the peaks in the original signal.
    ipks = zero_crossings_locations[:number_of_peaks] + ipk

    # Retrieve the signed max/min values at the peak locations.
    pks = input_signal[ipks]

    return pks, ipks


def signal_decomposition_algorithm(
    vertical_accelerarion_data, initial_sampling_frequency=100
):
    """
    Perform the Signal Decomposition algorithm on accelerometer data.

    Args:
        vertical_accelerarion_data (numpy.ndarray): Vertical Acceleration data.
        initial_sampling_frequency (float): Sampling frequency of the data.

    Returns:
        IC_seconds (numpy.ndarray): Detected IC (Initial Contact) timings in seconds.
        FC_seconds (numpy.ndarray): Detected FC (Foot-off Contact) timings in seconds.
    """
    # Error handling for invalid input data
    if not isinstance(vertical_accelerarion_data, np.ndarray):
        raise ValueError("vertical_acceleration_data must be a numpy.ndarray")

    if len(vertical_accelerarion_data.shape) < 1:
        raise ValueError("vertical_acceleration_data must have at least one dimension")

    if (
        not isinstance(initial_sampling_frequency, (int, float))
        or initial_sampling_frequency <= 0
    ):
        raise ValueError("The initial sampling frequency must be a positive float.")

    # Define the target sampling frequency for processing.
    target_sampling_frequency = 40

    # Resample and interpolate the vertical acceleration data to the target sampling frequency.
    smoothed_vertical_accelerarion_data = resample_interpolate(
        vertical_accelerarion_data,
        initial_sampling_frequency,
        target_sampling_frequency,
    )

    # Load filtering coefficients from a .mat file
    filtering_file = scipy.io.loadmat(mat_filter_coefficients_file)
    num = filtering_file["Num"][0, :]
    width_of_pad = 10000 * len(num)
    smoothed_vertical_accelerarion_data_padded = np.pad(
        smoothed_vertical_accelerarion_data, width_of_pad, mode="wrap"
    )

    # Remove 40Hz drift from the filtered data
    drift_removed_acceleration = highpass_filter(
        signal=smoothed_vertical_accelerarion_data_padded,
        sampling_frequency=target_sampling_frequency,
        method="iir",
    )

    # Filter data using the fir low-pass filter
    detrended_vertical_acceleration_signal = lowpass_filter(
        drift_removed_acceleration, method="fir"
    )

    # Remove the padding from the detrended signal
    detrended_vertical_acceleration_signal_lpf_rmzp = (
        detrended_vertical_acceleration_signal[
            width_of_pad
            - 1 : len(detrended_vertical_acceleration_signal)
            - width_of_pad
        ]
    )

    # Integrate the detrended acceleration signal
    det_ver_acc_sig_LPInt = (
        scipy.integrate.cumulative_trapezoid(
            detrended_vertical_acceleration_signal_lpf_rmzp, initial="0"
        )
        / target_sampling_frequency
    )

    # Perform the continuous wavelet transform on the filtered acceleration data
    smoothed_wavelet_result = apply_continuous_wavelet_transform(
        det_ver_acc_sig_LPInt,
        scales=9,
        desired_scale=9,
        wavelet="gaus2",
        sampling_frequency=target_sampling_frequency,
    )

    # Center the wavelet result around zero
    smoothed_wavelet_result = smoothed_wavelet_result - np.mean(smoothed_wavelet_result)
    smoothed_wavelet_result = np.array(smoothed_wavelet_result)

    # Apply max_peaks_between_zc funtion to find peaks and their locations.
    pks1, ipks1 = max_peaks_between_zc(smoothed_wavelet_result.T)

    # Calculate indx1 (logical indices of negative elements)
    indx1 = pks1 < 0

    # Extract IC (indices of negative peaks)
    indices_of_negative_peaks = ipks1[indx1]

    # Convert IC to seconds
    IC_seconds = indices_of_negative_peaks / target_sampling_frequency

    # Apply continuous wavelet transform
    accVLPIntCwt2 = apply_continuous_wavelet_transform(
        smoothed_wavelet_result,
        scales=8,
        desired_scale=8,
        wavelet="gaus2",
        sampling_frequency=target_sampling_frequency,
    )

    # Center the wavelet result around zero
    accVLPIntCwt2 = accVLPIntCwt2 - np.mean(accVLPIntCwt2)
    accVLPIntCwt2 = np.array(accVLPIntCwt2)

    # Apply max_peaks_between_zc funtion to find peaks and their locations.
    pks2, ipks2 = max_peaks_between_zc(accVLPIntCwt2.T)

    # Calculate indx2 (logical indices of negative elements)
    indx2 = pks2 > 0

    # Extract IC (indices of negative peaks)
    final_contact = ipks2[indx2]

    # Extract Foot-off Contact (FC) timings in seconds
    FC_seconds = final_contact / target_sampling_frequency
    
    # Plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(smoothed_wavelet_result, label='Smoothed Wavelet Transformed Vertical Acc')
    # plt.plot(accVLPIntCwt2, '--', label='Second Wavelet Transform')
    # plt.scatter(indices_of_negative_peaks, smoothed_wavelet_result[indices_of_negative_peaks], marker='o', color='red', label='IC Seconds')
    # plt.scatter(final_contact, accVLPIntCwt2[final_contact], marker='x', color='green', label='FC Seconds')
    # plt.title('Initial Contact (IC) and Final Contact (FC) Detection')
    # plt.xlabel('Samples')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    return IC_seconds, FC_seconds


# Function to classify activity levels based on accelerometer data
def classify_physical_activity(
    input_data,
    time_column_name="timestamp",
    sedentary_threshold=45,
    light_threshold=100,
    moderate_threshold=400,
    epoch_duration=5,
):
    """
    Classify activity levels based on processed Euclidean Norm Minus One (ENMO) values.

    Args:
        input_data (DataFrame): Input data with time index and accelerometer data (N, 3) for x, y, and z axes.
        time_column_name (str): Name of the index column.
        sedentary_threshold (float): Threshold for sedentary activity.
        light_threshold (float): Threshold for light activity.
        moderate_threshold (float): Threshold for moderate activity.
        epoch_duration (int): Duration of each epoch in seconds.

    Returns:
        DataFrame: Processed data including time, averaged ENMO values base on epoch length, activity levels represented with 0 or 1.
    """
    # Check if input_data is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input_data must be a pandas DataFrame.")

    # Check if threshold values are valid numeric types
    if not all(
        isinstance(threshold, (int, float))
        for threshold in [sedentary_threshold, light_threshold, moderate_threshold]
    ):
        raise ValueError("Threshold values must be numeric.")

    # Check if epoch_duration is a positive integer
    if not isinstance(epoch_duration, int) or epoch_duration <= 0:
        raise ValueError("Epoch_duration must be a positive integer.")

    # Group data by time in epochs and calculate the mean
    processed_data = input_data.groupby(pd.Grouper(freq=f"{epoch_duration}S")).mean()

    # Classify activity levels based on threshold values
    processed_data["sedentary"] = (processed_data["enmo"] < sedentary_threshold).astype(
        int
    )
    processed_data["light"] = (
        (sedentary_threshold <= processed_data["enmo"])
        & (processed_data["enmo"] < light_threshold)
    ).astype(int)
    processed_data["moderate"] = (
        (light_threshold <= processed_data["enmo"])
        & (processed_data["enmo"] < moderate_threshold)
    ).astype(int)
    processed_data["vigorous"] = (processed_data["enmo"] >= moderate_threshold).astype(
        int
    )

    # Reset the index for the resulting DataFrame
    processed_data.reset_index(inplace=True)

    # Return a DataFrame with the time, averaged ENMO, and classes of sedentary, light, moderate and vigorous shown with 1 or 0.
    return processed_data[
        [time_column_name, "enmo", "sedentary", "light", "moderate", "vigorous"]
    ]


# Function to plot results of the gait sequence detection algorithm
def gsd_plot_results(
    target_sampling_freq_Hz, detected_activity_signal, gait_sequences_
):
    """
    Plot the detected gait sequences.

    Args:
        target_sampling_freq_Hz (float) : Target sampling frequency.
        detected_activity_signal (np.array): Pre-processed acceleration signal.
        gait_sequences_ (pd.DataFrame): Detected gait sequences.

    Returns:
        plot
    """
    plt.figure(figsize=(22, 14))
    plt.plot(
        np.arange(len(detected_activity_signal)) / (60 * target_sampling_freq_Hz),
        detected_activity_signal,
        label="Pre-processed acceleration signal",
        color=cfg_colors["prep"][0],
    )
    plt.title("Detected gait sequences", fontsize=20)
    plt.xlabel("Time (minutes)", fontsize=20)
    plt.ylabel("Acceleration (g)", fontsize=20)

    # Fill the area between start and end times
    for index, sequence in gait_sequences_.iterrows():
        onset = sequence["onset"] / 60  # Convert to minutes
        end_time = (sequence["onset"] + sequence["duration"]) / 60  # Convert to minutes
        plt.axvline(onset, color=cfg_colors["raw"][1])
        plt.axvspan(onset, end_time, facecolor="grey", alpha=0.8)
    plt.legend(
        ["Pre-processed acceleration signal", "Gait onset", "Gait duration"],
        fontsize=20,
        loc="best",
    )
    plt.grid(visible=None, which="both", axis="both")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


# Function to plot results of the physical activity monitoring algorithm
def pam_plot_results(hourly_average_data, thresholds_mg):
    """
    Plots the hourly averaged ENMO for each day along with activity level thresholds.

    Args:
        hourly_average_data (pd.DataFrame): DataFrame containing hourly averaged ENMO values.
        thresholds_mg (dict): Dictionary containing threshold values for physical activity detection.
    """
    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    # Choose the 'turbo' colormap for coloring each day
    colormap = plt.cm.turbo

    # Plot thresholds
    ax.axhline(
        y=thresholds_mg.get("sedentary_threshold", 45),
        color="y",
        linestyle="--",
        label="Sedentary threshold",
    )
    ax.axhline(
        y=thresholds_mg.get("light_threshold", 100),
        color="g",
        linestyle="--",
        label="Light physical activity threshold",
    )
    ax.axhline(
        y=thresholds_mg.get("moderate_threshold", 400),
        color="r",
        linestyle="--",
        label="Moderate physical activity threshold",
    )

    # Plot each day data with a different color
    for i, date in enumerate(hourly_average_data.index):
        color = colormap(i)
        ax.plot(hourly_average_data.loc[date], label=str(date), color=color)

    # Customize plot appearance
    plt.xticks(range(24), [str(i).zfill(2) for i in range(24)])
    plt.xlabel("Time (h)", fontsize=16)
    plt.ylabel("ENMO (mg)", fontsize=16)
    plt.title("Hourly averaged ENMO for each day along with activity level thresholds")
    plt.legend(loc="upper left", fontsize=16)
    plt.grid(visible=None, which="both", axis="both")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()


# Function to estimate tilt angle
def tilt_angle_estimation(data, sampling_frequency_hz):
    """
    Estimate tilt angle using simple method with gyro data.

    Args:
        data (ndarray, DataFrame): Array or DataFrame containing gyro data.
        sampling_frequency_hz (float, int): Sampling frequency.

    Returns:
        tilt (ndarray): Tilt angle estimate (deg).
    """
    # Error handling for invalid input data
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    # Check if data is a numpy array
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array or pandas DataFrame")

    gyro_y = data[:, 1]

    # Integrate gyro data over time to estimate tilt
    tilt_angle = -np.cumsum(gyro_y) / sampling_frequency_hz

    return tilt_angle


#  Function for denoising using wavelet decomposition
def wavelet_decomposition(data, level, wavetype):
    """
    Denoise a signal using wavelet decomposition and reconstruction.

    Args:
        data (ndarray): Input signal to denoise.
        level (int): Order of wavelet decomposition.
        wavetype (str): Wavelet type to use.

    Returns:
        denoised_signal (ndarray): Denoised signal.
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data, wavetype, mode="smooth", level=level)

    # Zero out wavelet coefficients beyond specified order
    for i in range(1, len(coeffs)):
        if i != 0:  # Keep the first set of coefficients
            coeffs[i][:] = 0

    # Reconstruct signal from coefficients
    denoised_signal = pywt.waverec(coeffs, wavetype, mode="smooth")

    return denoised_signal


# Function for computing moving variance
def moving_var(data, window):
    """
    Compute the centered moving variance.

    Args:
        Data (int) : Data to take the moving variance on window
        Window size (int) : Window size for the moving variance.

    Returns
        m_var (numpy.ndarray) : Moving variance
    """

    # Initialize an array to store the moving variance
    m_var = np.zeros(data.shape)

    # Calculate the padding required
    pad = int(np.ceil(window / 2))

    # Define the shape and strides for creating rolling windows
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)

    # Create rolling windows from the input data
    rw_seq = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    # Compute the variance along the rolling windows and store it in m_var
    n = rw_seq.shape[0]
    m_var[pad : pad + n] = np.var(rw_seq, axis=-1, ddof=1)

    # Copy the variance values to the padding regions
    m_var[:pad], m_var[pad + n :] = m_var[pad], m_var[-pad - 1]

    return m_var


# Function to plot results of the gait sequence detection algorithm
def pham_plot_results(accel, gyro, postural_transitions_, sampling_freq_Hz):
    """
    Plot results of the gait sequence detection algorithm.

    Args:
        accel (ndarray): Array of acceleration data.
        gyro (ndarray): Array of gyroscope data.
        postural_transitions_ (DataFrame): DataFrame containing postural transition information.
        sampling_freq_Hz (float): Sampling frequency in Hertz.

    Returns:
        Plot postural transitions
    """
    # Figure
    fig = plt.figure(figsize=(21, 10))

    # Subplot 1: Acceleration data
    ax1 = plt.subplot(211)
    for i in range(3):
        ax1.plot(
            np.arange(len(accel)) / sampling_freq_Hz,
            accel[:, i],
        )
    for i in range(len(postural_transitions_)):
        onset = postural_transitions_["onset"][i]
        duration = postural_transitions_["duration"][i]
        ax1.axvline(x=onset, color="r")
        ax1.axvspan(onset, (onset + duration), color="grey")
    ax1.set_title("Detected Postural Transitions", fontsize=20)
    ax1.set_ylabel(f"Acceleration (g)", fontsize=14)
    ax1.set_xlabel(f"Time (sec)", fontsize=14)
    ax1.legend(
        ["Acc 1", "Acc 2", "Acc 3", "Event oset", "Event duration"],
        loc="upper right",
        fontsize=14,
    )
    ax1.set_ylim(-2, 2.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Subplot 2: Gyro data
    ax2 = plt.subplot(212)
    for i in range(3):
        ax2.plot(
            np.arange(len(gyro)) / sampling_freq_Hz,
            gyro[:, i],
        )
    for i in range(len(postural_transitions_)):
        onset = postural_transitions_["onset"][i]
        duration = postural_transitions_["duration"][i]
        ax2.axvline(x=onset, color="r")
        ax2.axvspan(onset, (onset + duration), color="grey")
    ax1.set_title("Detected Postural Transitions", fontsize=20)
    ax2.set_ylabel(f"Gyro (deg/s)", fontsize=14)
    ax2.set_xlabel(f"Time (sec)", fontsize=14)
    ax2.legend(
        ["Gyr 1", "Gyr 2", "Gyr 3", "Event oset", "Event duration"],
        loc="upper right",
        fontsize=14,
    )
    ax2.set_ylim(-200, 200)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    plt.show()


# Function to detect postural transitions based on stationary periods
def process_postural_transitions_stationary_periods(
    time,
    accel,
    gyro,
    stationary,
    tilt_angle_deg,
    sampling_period,
    sampling_freq_Hz,
    init_period,
    local_peaks,
):
    """
    Estimate orientation and analyze postural transitions based on sensor data.

    Args:
        time (ndarray): Array of timestamps.
        accel (ndarray): Array of accelerometer data (3D).
        gyro (ndarray): Array of gyroscope data (3D).
        stationary (ndarray): Array indicating stationary periods.
        tilt_angle_deg (ndarray): Array of tilt angle data.
        sampling_period (float): Sampling period in seconds.
        sampling_freq_Hz (float): Sampling frequency in Hz.
        init_period (float): Initialization period in seconds.
        local_peaks (ndarray): Array of indices indicating local peaks.

    Returns:
        tuple: A tuple containing:
            time_pt (list): List of peak times.
            pt_type (list): List of postural transition types.
            pt_angle (list): List of postural transition angles.
            duration (list): List of postural transition durations.
            flexion_max_vel (list): List of maximum flexion velocities.
            extension_max_vel (list): List of maximum extension velocities.
    """
    # If there is enough stationary data, perform sensor fusion using accelerometer and gyro data
    # Initialize quaternion array for orientation estimation
    quat = np.zeros((len(time), 4))

    # Initial convergence: Update the quaternion using the mean accelerometer values over a certain period
    # This helps in initializing the orientation for accurate estimation
    index_sel = np.arange(0, np.where(time >= time[0] + init_period)[0][0] + 1)
    mean_accel = np.mean(accel[index_sel], axis=0)
    quat[0] = quaternion.rotm2quat(np.eye(3) + quaternion.axang2rotm(mean_accel))

    # Update the quaternion for all data points
    for t in range(1, len(time)):
        # Calculate the rotation matrix from gyroscope data
        dt = time[t] - time[t - 1]
        ang_velocity = gyro[t] * dt
        delta_rot = quaternion.axang2rotm(ang_velocity)

        # Update the quaternion based on the rotation matrix
        quat[t] = quaternion.quatmultiply(quat[t - 1], quaternion.rotm2quat(delta_rot))

        # Normalize the quaternion to avoid drift
        quat[t] = quaternion.quatnormalize(quat[t])

    # Analyze gyro data to detect peak velocities and directional changes
    # Zero-crossing method is used to define the beginning and the end of a PT in the gyroscope signal
    iZeroCr = np.where((gyro[:, 1][:-1] * gyro[:, 1][1:]) < 0)[0]

    # Calculate the difference between consecutive values
    gyrY_diff = np.diff(gyro[:, 1])

    # Beginning of a PT was defined as the first zero crossing point of themedio-lateral angular
    # velocity (gyro[:,1]) on the left side of the PT event, with negative slope.
    # Initialize left side indices with ones
    ls = np.ones_like(local_peaks)

    # Initialize right side indices with length of gyro data
    # rs = len(gyro[:,1]) * np.ones_like(local_peaks)
    rs = np.full_like(local_peaks, len(gyro[:, 1]))
    for i in range(len(local_peaks)):
        # Get the index of the current local peak
        pt = local_peaks[i]

        # Calculate distances to all zero-crossing points relative to the peak
        dist2peak = iZeroCr - pt

        # Extract distances to zero-crossing points on the left side of the peak
        dist2peak_ls = dist2peak[dist2peak < 0]

        # Extract distances to zero-crossing points on the right side of the peak
        dist2peak_rs = dist2peak[dist2peak > 0]

        # Iterate over distances to zero-crossing points on the left side of the peak (in reverse order)
        for j in range(len(dist2peak_ls) - 1, -1, -1):
            # Check if slope is down and the left side not too close to the peak (more than 200ms)
            if gyrY_diff[pt + dist2peak_ls[j]] < 0 and -dist2peak_ls[j] > 25:
                # Store the index of the left side
                ls[i] = pt + dist2peak_ls[j]
                break

    # Further analysis to distinguish between different types of postural transitions (sit-to-stand or stand-to-sit)
    # Rotate body accelerations to Earth frame
    acc = quaternion.rotm2quat(
        np.column_stack((accel[:, 0], accel[:, 1], accel[:, 2])), quat
    )

    # Remove gravity from measurements
    acc -= np.array([[0, 0, 1]] * len(time))

    # Convert acceletion data to m/s^2
    acc *= 9.81

    # Calculate velocities
    vel = np.zeros_like(acc)

    # Iterate over time steps
    for t in range(1, len(vel)):
        # Integrate acceleration to calculate velocity
        vel[t, :] = vel[t - 1, :] + acc[t, :] * sampling_period
        if stationary[t] == 1:
            # Force zero velocity when stationary
            vel[t, :] = [0, 0, 0]

    # Compute and remove integral drift
    velDrift = np.zeros_like(vel)

    # Indices where stationary changes to non-stationary
    activeStart = np.where(np.diff(stationary) == -1)[0]

    # Indices where non-stationary changes to stationary
    activeEnd = np.where(np.diff(stationary) == 1)[0]
    if activeStart[0] > activeEnd[0]:
        # Ensure start from index 0 if starts non-stationary
        activeStart = np.insert(activeStart, 0, 0)

    if activeStart[-1] > activeEnd[-1]:
        # Ensure last segment ends properly
        activeEnd = np.append(activeEnd, len(stationary))
    for i in range(len(activeEnd)):
        # Calculate drift rate
        driftRate = vel[activeEnd[i] - 1] / (activeEnd[i] - activeStart[i])

        # Enumerate time steps within the segment
        enum = np.arange(1, activeEnd[i] - activeStart[i] + 1)

        # Calculate drift for each time step
        drift = np.column_stack(
            (enum * driftRate[0], enum * driftRate[1], enum * driftRate[2])
        )

        # Store the drift for this segment
        velDrift[activeStart[i] : activeEnd[i], :] = drift

    # Remove integral drift from velocity
    vel -= velDrift

    # Compute translational position
    pos = np.zeros_like(vel)

    # Iterate over time steps
    for t in range(1, len(pos)):
        # Integrate velocity to yield position
        pos[t, :] = pos[t - 1, :] + vel[t, :] * sampling_period

    # Estimate vertical displacement and classify as actual PTs or Attempts
    # Calculate vertical displacement
    disp_z = pos[rs, 2] - pos[ls, 2]

    # Initialize flag for actual PTs
    pt_actual_flag = np.zeros_like(local_peaks)

    for i in range(len(disp_z)):
        # Displacement greater than 10cm and less than 1m
        if 0.1 < abs(disp_z[i]) < 1:
            # Flag as actual PT if displacement meets criteria
            pt_actual_flag[i] = 1

    # Initialize list for PT types
    pt_type = []

    # Distinguish between different types of postural transitions
    for i in range(len(local_peaks)):
        if pt_actual_flag[i] == 1:
            if disp_z[i] == 0:
                pt_type.append("NA")
            elif disp_z[i] > 0:
                pt_type.append("sit to stand")
            else:
                pt_type.append("stand to sit")
        else:
            pt_type.append("NA")

    # Calculate maximum flexion velocity and maximum extension velocity
    flexion_max_vel = np.zeros_like(local_peaks)
    extension_max_vel = np.zeros_like(local_peaks)
    for i in range(len(local_peaks)):
        flexion_max_vel[i] = max(abs(gyro[:, 1][ls[i] : local_peaks[i]]))
        extension_max_vel[i] = max(abs(gyro[:, 1][local_peaks[i] : rs[i]]))

    # Calculate PT angle
    pt_angle = np.abs(tilt_angle_deg[local_peaks] - tilt_angle_deg[ls])
    if ls[0] == 0:
        # Adjust angle for the first PT if necessary
        pt_angle[0] = np.abs(tilt_angle_deg[local_peaks[0]] - tilt_angle_deg[rs[0]])

    # Calculate duration of each PT
    duration = (rs - ls) / sampling_freq_Hz

    # Convert peak times to integers
    time_pt = time[local_peaks]

    # Initialize PTs list
    # i.e., the participant was considered to perform a complete standing up or sitting down movement
    PTs = [
        [
            "Time[s]",
            "Type",
            "Angle[°]",
            "Duration[s]",
            "Max flexion velocity[°/s]",
            "Max extension velocity[°/s]",
            "Vertical displacement[m]",
        ]
    ]

    # Initialize Attempts list
    # i.e., the participant was considered not to perform a complete PT, e.g., forward and backwards body motion
    Attempts = [
        [
            "Time[s]",
            "Type",
            "Angle[°]",
            "Duration[s]",
            "Max flexion velocity[°/s]",
            "Max extension velocity[°/s]",
            "Vertical displacement[m]",
        ]
    ]

    # Iterate over detected peaks
    for i in range(len(local_peaks)):
        if pt_actual_flag[i] == 1:
            PTs.append(
                [
                    time_pt[i],
                    pt_type[i],
                    pt_angle[i],
                    duration[i],
                    flexion_max_vel[i],
                    extension_max_vel[i],
                    disp_z[i],
                ]
            )  # Append PT details to PTs list
        else:
            Attempts.append(
                [
                    time_pt[i],
                    pt_type[i],
                    pt_angle[i],
                    duration[i],
                    flexion_max_vel[i],
                    extension_max_vel[i],
                    disp_z[i],
                ]
            )  # Append PT details to Attempts list

    # Extract postural transition information from PTs
    time_pt = [pt[0] for pt in PTs[1:]]
    pt_type = [pt[1] for pt in PTs[1:]]
    pt_angle = [pt[2] for pt in PTs[1:]]
    duration = [pt[3] for pt in PTs[1:]]
    flexion_max_vel = [pt[4] for pt in PTs[1:]]
    extension_max_vel = [pt[5] for pt in PTs[1:]]

    # Return the necessary outputs
    return time_pt, pt_type, pt_angle, duration, flexion_max_vel, extension_max_vel
