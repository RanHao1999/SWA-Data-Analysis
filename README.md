# SWA-Data-Analysis
Codes for analyzing Solar Orbiter Solar Wind Analyzer (SWA) data.

For ion:
Gausssian Mixture Model (GMM) is employed to separate protons and alpha particles in the measurements.
Please use the GMM_3component_tutorial.ipynb to make sure that the first timeslice can be well separated.
The user needs to manually set the index between alpha and proton, the variable is "dividing_idx".
Then, the user can run GMM_3component.py for a long interval.
These codes are well-commented. Enjoy!

====================================================================================================

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

====================================================================================================

If you want to have a higher efficiency and are happy to sacrifice a bit accuracy, we managed to parallelised the code.
Block-wise warm-start for parallel GMM clustering

-----------------------------------------------------------------------------
To enable parallelisation while keeping temporal continuity, we process the
time series in blocks (e.g. N = 15 slices ≈ 1 min at 4 s cadence).

   1. First block:
        - Fit the first slice sequentially with a generic initial guess.
        - Use its converged parameters as the common initial guess for all
          other slices in the block, which are then fitted in parallel.

   2. Subsequent blocks:
        - Take the final GMM solution from the previous block.
        - Use it as the initial guess for the entire new block.
        - Fit all slices in the block in parallel.

Thus, blocks depend sequentially on each other, but slices within each block
 can be processed independently across multiple cores.
 
 -----------------------------------------------------------------------------

 Go to GMM_3components_parallelised.py
 Set parameters following GMM_3components.py, but also set:
 
    # block length in seconds
    block_length = 60  # 1 minutes, 15 timeslices
    block_size = block_length // 4  # since data is at 4s resolution
    
    # Number of parallel processes, for maximum efficiency, we recommend it to be able to be evenly divided by the block size.
    n_processes = 15 

====================================================================================================

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




