#%% calculate conversion rate of the pupil size from area in pixels to diameter
import numpy as np

pixel_size = 2.4*10**-6
focal_length = 16*10**-3

distances = np.array([581, 588, 586, 599, 589]) # distance between camera and eyes
pupil_area_pix = np.array([[7277.2724777, 6543.82448724, 6100.47891878, 5650.69485044, 4432.27097012],
  [4762.25010773, 4792.81050739 ,4406.86177249, 4186.407594,   4028.83946451],
  [4729.40655033, 2911.96121051, 3863.90175481, 3654.97422491, 3838.9650351 ],
  [          np.nan, 4998.5918674,  5176.71187517, 4896.23716166, 4671.88410215],
  [9226.96789759, 7337.59041274, 6558.01580546, 6444.46917251, 5695.41671914]])


pixel_size_d = pixel_size*distances/focal_length # pixel size at distance d

scaling_factor = np.mean(pixel_size_d**2)

pupil_areas_mm = np.zeros((5,5))
pupil_diameters = np.zeros((5,5))
for index, size_d in enumerate(pixel_size_d):
    areas = pupil_area_pix[index]
    pupil_areas = areas * size_d**2
    pupil_areas_mm[index,:] = pupil_areas
    pupil_diameters[index,:] = 2 * np.sqrt(pupil_areas/np.pi)
    
pupil_diam_mm = 2*np.sqrt((scaling_factor*pupil_area_pix)/np.pi)
print(pupil_diam_mm)