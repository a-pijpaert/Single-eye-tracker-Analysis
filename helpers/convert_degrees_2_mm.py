import numpy as np

def convert_degrees_2_mm(angle: float, viewing_distance: float) -> float:
    """
    Converts an angle in degrees to an equivalent distance in millimeters (mm), based on the viewing distance.

    The function first converts the given angle from degrees to radians. Then, it calculates the distance in millimeters
    using the tangent of the angle and the viewing distance.

    Parameters:
    angle (float): The angle in degrees. This angle is between the line of sight and the point where the line of sight intersects the display.
    viewing_distance (float): The distance from the viewer to the display, measured in millimeters (mm).

    Returns:
    float: The distance in millimeters on the display that corresponds to the specified angle at the given viewing distance.

    Example:
    # Convert a 5-degree angle to millimeters at a viewing distance of 600mm:
    distance_mm = convert_degrees_2_mm(5, 600)

    Note:
    This function assumes a flat display and may not be accurate for curved displays.
    """
    
    angle_radians = np.radians(angle)
    distance_mm = viewing_distance * np.tan(angle_radians)

    return distance_mm
