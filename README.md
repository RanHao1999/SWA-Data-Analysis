# SWA-Data-Analysis
Codes for analyzing Solar Orbiter Solar Wind Analyzer (SWA) data.

For Electrons:
EAS1 and EAS2 are separately processed, with low-energy range (< 10 eV, highly contaminated by photo- and secondary- electrons) replaced by a bi-Maxwellian fitting of the core (10 eV to 70 eV).

For ion:
Gausssian Mixture Model (GMM) is employed to separate protons and alpha particles in the measurements.
Please use the GMM_3component_tutorial.ipynb to make sure that the first timeslice can be well separated.
The user needs to manually set the index between alpha and proton, the variable is "dividing_idx".
Then, the user can run GMM_3component.py for a long interval.
These codes are well-commented. Enjoy!

Steps for using the GMM codes:

Let’s say you want to run the GMM from t1 to t2.

1.Go to GMM_Hao_Tutorial.ipynb,
Set “yymmdd” and “timeslice” to t1,

Manually set “dividing_idx” (the separation velocity index between alpha and proton),

Run the rest of the code

2.Go to GMM_3components.py
In the main() function: Set t_start as t1 + 4s,

Set t_end as t2,

Run the code



Folder structure:
  - main
    - codes in the main folder
    - data
      - SO
        - yymmdd
          swa-pas-3d, mag_srf, swa-pas-grnd-mom, swa-pas-vdf
    - result
      - SO
        - yymmdd
          - Particles
            - Ions
              - hhmmss....


Any questions, feel free to contact me: hao.ran.24@ucl.ac.uk




