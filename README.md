# SWA-Data-Analysis
Codes for analyzing Solar Orbiter Solar Wind Analyzer (SWA) data.

For Electrons:
EAS1 and EAS2 are separately processed, with low-energy range (< 10 eV, highly contaminated by photo- and secondary- electrons) replaced by a bi-Maxwellian fitting of the core (10 eV to 70 eV).

For ion:
Gausssian Mixture Model (GMM) is employed to separate protons and alpha particles in the measurements.
Please use the GMM_3component_initial.ipynb to make sure that the first timeslice can be well separated.
The user needs to manually set the index between alpha and proton, the variable is "dividing_idx".
Then, the user can run GMM_3component.py to run for a long interval. Please set the initial input as the "SolarWindParticle" saved from GMM_3component_initial.ipynb.
These codes are well-commented. Enjoy!


Any questions, feel free to contact me: hao.ran.24@ucl.ac.uk




