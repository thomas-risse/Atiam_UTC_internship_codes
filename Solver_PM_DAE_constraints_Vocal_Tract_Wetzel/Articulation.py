import numpy as np


class VowelArticulations:
    """Class used to generate vocal tract profiles corresponding to different
    vowels. Profiles are base on "MRI-based vocal tract representations for 
    the three-dimensional finite element synthesis of diphthongs" from
    Marc Arnela, Saeed Dabbaghchian, Oriol Guasch and Olov Engwall.
    """
    def __init__(self, L0) -> None:
        self.articulations = np.array(["a", "i", "u"])
        # Abcissae from the glottis
        self.points = dict(
            {
                "a": 1e-2
                * np.array(
                    [
                        0,
                        0.63,
                        1.29,
                        1.92,
                        2.62,
                        3.09,
                        3.53,
                        3.95,
                        4.38,
                        4.81,
                        5.24,
                        5.65,
                        6.07,
                        6.5,
                        6.92,
                        7.35,
                        7.78,
                        8.22,
                        8.68,
                        9.15,
                        9.64,
                        10.13,
                        10.6,
                        11.09,
                        11.57,
                        12.05,
                        12.51,
                        13.03,
                        13.46,
                        13.88,
                        14.3,
                        14.73,
                        15.16,
                        15.58,
                        16,
                        16.53,
                        16.99,
                        17.49,
                        17.94,
                        18.38,
                    ]
                ),
                "i": 1e-2
                * np.array(
                    [
                        0,
                        0.62,
                        1.26,
                        1.89,
                        2.72,
                        3.07,
                        3.52,
                        3.94,
                        4.36,
                        4.79,
                        5.21,
                        5.77,
                        6.18,
                        6.6,
                        7.02,
                        7.44,
                        7.86,
                        8.3,
                        8.75,
                        9.24,
                        9.71,
                        10.19,
                        10.66,
                        11.11,
                        11.56,
                        12.01,
                        12.44,
                        12.86,
                        13.28,
                        13.72,
                        14.17,
                        14.63,
                        15.08,
                        15.52,
                        15.96,
                        16.39,
                        16.83,
                        17.25,
                        17.67,
                        18.08,
                    ]
                ),
                "u": 1e-2
                * np.array(
                    [
                        0,
                        0.67,
                        1.36,
                        2.02,
                        2.72,
                        3.19,
                        3.68,
                        4.15,
                        4.63,
                        5.1,
                        5.67,
                        6.14,
                        6.61,
                        7.08,
                        7.56,
                        8.04,
                        8.55,
                        9.08,
                        9.63,
                        10.19,
                        10.77,
                        11.36,
                        11.97,
                        12.54,
                        13.1,
                        13.65,
                        14.16,
                        14.66,
                        15.19,
                        15.66,
                        16.14,
                        16.62,
                        17.11,
                        17.6,
                        18.1,
                        18.56,
                        19.03,
                        19.49,
                        19.96,
                        20.42,
                    ]
                ),
            }
        )
        # Cross sections in m^2
        self.cross_sections = dict(
            {
                "a": 1e-4
                * np.array(
                    [
                        0.2,
                        0.5,
                        0.3,
                        0.36,
                        2.67,
                        2.96,
                        2.73,
                        2.38,
                        1.95,
                        1.59,
                        2.12,
                        2.11,
                        1.8,
                        1.47,
                        1.28,
                        1.04,
                        1.17,
                        1.37,
                        1.65,
                        2.32,
                        2.76,
                        2.53,
                        2.89,
                        2.31,
                        2.77,
                        3.42,
                        4.14,
                        7.17,
                        9.19,
                        11.81,
                        12.63,
                        12.42,
                        13.09,
                        12.71,
                        12.25,
                        13.73,
                        12.65,
                        9.61,
                        6.52,
                        4.81,
                    ]
                ),
                "i": 1e-4
                * np.array(
                    [
                        0.3,
                        0.53,
                        0.28,
                        0.38,
                        2.87,
                        3.55,
                        3.88,
                        3.98,
                        3.93,
                        3.73,
                        3.66,
                        6.53,
                        6.64,
                        6.82,
                        6.97,
                        7.27,
                        7.51,
                        7.46,
                        7.5,
                        7.6,
                        6.46,
                        5.31,
                        4.62,
                        4.22,
                        2.78,
                        1.59,
                        0.76,
                        0.38,
                        0.32,
                        0.35,
                        0.37,
                        0.47,
                        0.47,
                        0.54,
                        0.51,
                        0.6,
                        1,
                        1.29,
                        2.4,
                        3.06,
                    ]
                ),
                "u": 1e-4
                * np.array(
                    [
                        0.15,
                        0.22,
                        0.05,
                        0.14,
                        1.65,
                        2.25,
                        2.28,
                        2.1,
                        1.71,
                        1.75,
                        4.06,
                        4.51,
                        3.95,
                        3.19,
                        2.62,
                        2.18,
                        1.7,
                        1.54,
                        1.8,
                        1.82,
                        1.56,
                        1.15,
                        1.31,
                        0.77,
                        0.56,
                        0.91,
                        1.77,
                        2.94,
                        5.13,
                        6.22,
                        7.25,
                        7.44,
                        7.37,
                        6.13,
                        3.85,
                        2.33,
                        1.09,
                        0.42,
                        0.15,
                        0.31,
                    ]
                ),
            }
        )
        # Correspondind heights in m
        self.L0 = L0
        self.populate_heights()

    def populate_heights(self):
        self.heights = self.cross_sections
        for key in self.heights.keys():
            self.heights[key] = self.heights[key] / self.L0

    def get_height(self, xs, vowel):
        """Returns the heights computed at points xs using linear interpolation.

        Args:
            xs (array): points
            vowel (string): name of the vowel

        Return:
            array: corresponding heights
        """
        points = self.points[vowel]
        h_points = self.heights[vowel]
        indexes = np.searchsorted(points, xs)
        heights = h_points[indexes - 1] + (
            h_points[indexes] - h_points[indexes - 1]
        ) * (xs - points[indexes - 1]) / (
            points[indexes] - points[indexes - 1]
        )
        return heights

    def generate_profiles(self, N_tract):
        """Generates profiles for vowels given a number of tracts.

        Args:
            N_tract (int): Number of tracts of the model.
        """
        self.profile = {"N tract": N_tract}
        for key in self.points.keys():
            points = self.points[key]
            h_points = self.heights[key]
            limit_points = np.linspace(0, np.max(points), N_tract + 1)
            indexes = np.searchsorted(points, limit_points)
            mean_values = np.zeros((N_tract))
            h_limit_points = self.get_height(limit_points, key)
            means_left = (h_limit_points[:-1] + h_points[indexes[:-1]]) / 2
            means_right = (h_points[indexes[1:] - 1] + h_limit_points[1:]) / 2
            means_middle = (
                h_points[indexes[:-1]] + h_points[indexes[:-1] + 1]
            ) / 2
            mean_values = means_left + means_middle + means_right
            self.profile[key] = mean_values
