# Import libraries
from typing import Optional, Literal, Union, TypeVar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from ngmt.utils import preprocessing
from ngmt.config import cfg_colors

Self = TypeVar("Self", bound="ParaschivIonescuInitialContactDetection")

class ParaschivIonescuInitialContactDetection:
    """
    This Paraschiv-Ionescu initial contact detection algorithm identifies initial contact in accelerometer data
    collected from a low back IMU sensor. The purpose of algorithm is to identify and characterize initial contacts
    within walking bouts.

    The algorithm takes accelerometer data as input, and the vertical acceleration component, and processes each
    specified gait sequence independently. The signal is first detrended and then low-pass filtered. The resulting
    signal is numerically integrated and differentiated using a Gaussian continuous wavelet transformation. The
    initial contact (IC) events are identified as the positive maximal peaks between successive zero-crossings.
    In addition, final contacts (FC) events are also identified as times of the maxima of the signal obtained from 
    a further continuous wavelet transformation differentiation. 

    Finally, initial contacts and final contacts information are provided as a DataFrame with columns `onset`, `event_type`, and
    `tracking_systems`.

    Methods:
        detect(data, gait_sequences, sampling_freq_Hz):
            Detects initial contacts on the accelerometer signal.

    Examples:
        Find initial contacts based on the detected gait sequence

        >>> icd = ParaschivIonescuInitialContactDetection()
        >>> icd = icd.detect(data=acceleration_data, sampling_freq_Hz=100)
        >>> print(icd.initial_contacts_)
                onset   event_type       duration   tracking_systems
            0   5       initial contact  0          SU
            1   5.6     final   contact  0          SU

        Calculate temporal parameters

        >>> icd.temporal_parameters()
        >>> print(icd.parameters_)
                stride time [s]  swing time [s]  stance time [s]
            0   1.2              0.4             0.8
            1   1.1              0.3             0.8

    References:
        [1] Paraschiv-Ionescu et al. (2019). Locomotion and cadence detection using a single trunk-fixed accelerometer...

        [2] Paraschiv-Ionescu et al. (2020). Real-world speed estimation using single trunk IMU: methodological challenges...

        [3] McCamley et al. (2012). An enhanced estimate of initial contact and final contact instants of time using...
    """

    def __init__(
        self,
    ):
        """
        Initializes the ParaschivIonescuInitialContactDetection instance.
        """
        self.initial_contacts_ = None

    def detect(
        self,
        data: pd.DataFrame,
        sampling_freq_Hz: float,
        v_acc_col_name: str,
        gait_sequences: Optional[pd.DataFrame] = None,
        dt_data: Optional[pd.Series] = None,
        tracking_system: Optional[str] = None,
    ) -> Self:
        """
        Detects initial contacts based on the input accelerometer data.

        Args:
            data (pd.DataFrame): Input accelerometer data (N, 3) for x, y, and z axes.
            sampling_freq_Hz (float): Sampling frequency of the accelerometer data.
            v_acc_col_name (str): The column name that corresponds to the vertical acceleration.
            gait_sequences (pd.DataFrame, optional): A dataframe of detected gait sequences. If not provided, the entire acceleration time series will be used for detecting initial contacts.
            dt_data (pd.Series, optional): Original datetime in the input data. If original datetime is provided, the output onset will be based on that.
            tracking_system (str, optional): Tracking system the data is from to be used for events df. Default is None.

        Returns:
            ParaschivIonescuInitialContactDetection: Returns an instance of the class.
                The initial contacts information is stored in the 'initial_contacts_' attribute,
                which is a pandas DataFrame in BIDS format with the following columns:
                    - onset: initial contact and final contact events onset
                    - event_type: Type of the event (initial contact or final contact).
                    - tracking_system: Tracking systems used the events are derived from.
        """
        # Check if data is empty
        if data.empty:
            self.initial_contacts_ = pd.DataFrame()
            return self  # Return without performing further processing

        # check if dt_data is a pandas Series with datetime values
        if dt_data is not None and (
            not isinstance(dt_data, pd.Series)
            or not pd.api.types.is_datetime64_any_dtype(dt_data)
        ):
            raise ValueError("dt_data must be a pandas Series with datetime values")

        # check if tracking_system is a string
        if tracking_system is not None and not isinstance(tracking_system, str):
            raise ValueError("tracking_system must be a string")

        # check if dt_data is provided and if it is a series with the same length as data
        if dt_data is not None and len(dt_data) != len(data):
            raise ValueError("dt_data must be a series with the same length as data")

        # Extract vertical accelerometer data using the specified index
        acc_vertical = data[v_acc_col_name]

        # Initialize an empty list to store the processed output
        processed_output = []

        # Initialize an empty list to store all onsets
        all_onsets = []
        
        # Initialize an empty list to store event types
        event_types = []

        # Process each gait sequence
        if gait_sequences is None:
            gait_sequences = pd.DataFrame(
                {"onset": [0], "duration": [len(data) / sampling_freq_Hz]}
            )
        for _, gait_seq in gait_sequences.iterrows():
            # Calculate start and stop indices for the current gait sequence
            start_index = int(sampling_freq_Hz * gait_seq["onset"])
            stop_index = int(sampling_freq_Hz * (gait_seq["onset"] + gait_seq["duration"]))
            accv_gait_seq = acc_vertical[start_index:stop_index + 2].to_numpy()

            try:
                # Perform Signal Decomposition Algorithm for Initial Contacts (ICs) and Final Contacts (FCs)
                initial_contacts_rel, final_contacts_rel = preprocessing.signal_decomposition_algorithm(
                    accv_gait_seq, sampling_freq_Hz
                )
                initial_contacts = gait_seq["onset"] + initial_contacts_rel
                final_contacts = gait_seq["onset"] + final_contacts_rel

                # Combine and sort initial and final contacts
                all_contacts = np.sort(np.concatenate((initial_contacts, final_contacts)))
                all_event_types = ["initial contact"] * len(initial_contacts) + ["final contact"] * len(final_contacts)
                all_event_types = [x for _, x in sorted(zip(np.concatenate((initial_contacts, final_contacts)), all_event_types))]

                # Ensure first and last elements are initial contacts
                if all_event_types[0] != "initial contact":
                    all_contacts = all_contacts[1:]
                    all_event_types = all_event_types[1:]
                if all_event_types[-1] != "initial contact":
                    all_contacts = all_contacts[:-1]
                    all_event_types = all_event_types[:-1]

                gait_seq["IC"] = [all_contacts[i] for i in range(len(all_contacts)) if all_event_types[i] == "initial contact"]
                gait_seq["FC"] = [all_contacts[i] for i in range(len(all_contacts)) if all_event_types[i] == "final contact"]

                # Append onsets and event types to the lists
                all_onsets.extend(all_contacts)
                event_types.extend(all_event_types)

            except Exception as e:
                print(
                    "Signal decomposition algorithm did not run successfully. Returning an empty vector of initial contacts"
                )
                print(f"Error: {e}")
                initial_contacts = []
                gait_seq["IC"] = []

            # Append the information to the processed_output list
            processed_output.append(gait_seq)

        # Check if processed_output is not empty
        if not processed_output:
            print("No initial contacts detected.")
            return pd.DataFrame()

        # Create a DataFrame from the processed_output list
        initial_contacts_ = pd.DataFrame(processed_output)

        # Create a BIDS-compatible DataFrame with all onsets
        self.initial_contacts_ = pd.DataFrame(
            {
                "onset": all_onsets,
                "event_type": event_types,
                "duration": 0,
                "tracking_systems": tracking_system,
            }
        )

        # If original datetime is available, update the 'onset' column
        if dt_data is not None:
            valid_indices = [
                index
                for index in self.initial_contacts_["onset"]
                if index < len(dt_data)
            ]
            invalid_indices = len(self.initial_contacts_["onset"]) - len(valid_indices)

            if invalid_indices > 0:
                print(f"Warning: {invalid_indices} invalid index/indices found.")

            # Only use valid indices to access dt_data
            valid_dt_data = dt_data.iloc[valid_indices]

            # Update the 'onset' column
            self.initial_contacts_["onset"] = valid_dt_data.reset_index(drop=True)

        return self


    def temporal_parameters(self: Self) -> Self:
        """
        Calculate temporal parameters based on detected initial and final contacts.

        Returns:
            ParaschivIonescuInitialContactDetection: Returns an instance of the class.
                The temporal parameters information is stored in the 'parameters_' attribute,
                which is a pandas DataFrame with the following columns:
                    - stride time [s]: Time for one stride [s].
                    - swing time [s]: Time for the swing phase [s].
                    - stance time [s]: Time for the stance phase [s].
        """
        if self.initial_contacts_ is None or self.initial_contacts_.empty:
            raise ValueError("No initial contacts detected. Please run the detect method first.")

        # Extract initial contacts and final contacts
        ic_onsets = self.initial_contacts_.loc[self.initial_contacts_["event_type"] == "initial contact", "onset"].values
        fc_onsets = self.initial_contacts_.loc[self.initial_contacts_["event_type"] == "final contact", "onset"].values

        # Ensure that there are enough IC and FC events to compute parameters
        if len(ic_onsets) < 3 or len(fc_onsets) < 2:
            raise ValueError("Not enough initial and final contacts to calculate temporal parameters.")

        # Calculate stride time, stance time, and swing time based on the provided formulas
        stride_time = ic_onsets[2:] - ic_onsets[:-2]
        stance_time = fc_onsets[1:] - ic_onsets[:-2]
        swing_time = stride_time - stance_time

        # Create a DataFrame with the calculated temporal parameters
        self.parameters_ = pd.DataFrame({
            "stride time [s]": stride_time,
            "swing time [s]": swing_time,
            "stance time [s]": stance_time
        })

        # Set the index name to 'stride id'
        self.parameters_.index.name = "stride id"

        return self