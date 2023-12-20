"""Base classes for datasets and algorithms."""

# See: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html#restricted-keyword-list-for-channel-type
VALID_CHANNEL_TYPES = {
    "ACCEL",
    "ANGACCEL",
    "BARO",
    "GYRO",
    "JNTANG",
    "LATENCY",
    "MAGN",
    "MISC",
    "ORNT",
    "POS",
    "VEL",
}

# See: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html#restricted-keyword-list-for-channel-component
VALID_COMPONENT_TYPES = {"x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "n/a"}
