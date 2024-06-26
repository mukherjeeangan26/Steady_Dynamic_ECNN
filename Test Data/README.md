# Data Description

The excel spreadsheets provided here include steady-state and dynamic datasets for developing the ECNN model of the 
lumped superheater / heat exchanger (HX) system using different types of noise characterizations in training data.

## Steady-State HX Data

This spreadsheet contains 400 steady-state input-output datasets for the HX system. The model inputs are represented 
by the mass flowrates (m_FG, m_St), inlet temperatures (T_FG_in, T_St_in), and inlet pressures (P_FG_in, P_St_in) of 
flue gas and steam, respectively. The model outputs are denoted by the outlet temperatures of steam (T_St_out) and 
flue gas (T_FG_out) streams. Mass is trivially conserved in this system, i.e., the mass flowrates of both streams
remain the same across the system.

The .xlsx file contains 4 tabs in total. The last tab contains a block-oriented representation of the HX system under 
consideration. The first three tabs contain steady-state data for the different types of noise / uncertainties added 
to the simulated data to generate training data, i.e., no noise (where truth and measurements are the same), constant 
bias with an additional zero-mean Gaussian noise, and random bias with an additional Gaussian noise distribution, respectively.

Note that the base (truth) case, i.e., the tab named 'NoNoise' represents the dataset obtained from simulation of an appropriate
first-principles model. It was also observed that the difference in heat (energy) balance for the 'True Data', i.e., the data 
containing no noise was not exactly zero in megawatts (MW) scale with respect to the superheater system. The corresponding difference
in Q (MW) is also provided in the excel spreadsheet with respect to the true data obtained by simulation of the first-principles model.
Although the specific formulation of the training and forward problems remain independent of the specfic types of constraints imposed, 
in this work, it is desired to see whether the ECNN models converge at the 'True Data' considered even when trained against noisy transient 
data. Therefore, the corresponding constraint equations are modified to arrive at the Delta_Q (MW) values calculated with respect to the 'True Data'.

## Dynamic HX Data

This spreadsheet contains dynamic time-series (total duration of around 6000 time steps) input-output datasets for 
the HX system. The model inputs are represented by the mass flowrates (m_FG, m_St), inlet temperatures (T_FG_in, T_St_in), 
and inlet pressures (P_FG_in, P_St_in) of flue gas and steam, respectively. The model outputs are denoted by the outlet 
temperatures of steam (T_St_out) and flue gas (T_FG_out) streams. Mass is trivially conserved in this system, i.e., the 
mass flowrates of both streams remain the same across the system.

The .xlsx file contains 4 tabs in total. The last tab contains a block-oriented representation of the HX system under 
consideration. The first three tabs contain dynamic data for the different types of noise / uncertainties added to the 
simulated data to generate training data, i.e., no noise (where truth and measurements are the same), time-invariant bias 
with an additional zero-mean Gaussian noise, and time-varying bias with an additional Gaussian noise distribution, respectively. 

Note that the base (truth) case, i.e., the tab named 'NoNoise' represents the dataset obtained from simulation of an appropriate
first-principles model. It was also observed that the difference in heat (energy) balance for the 'True Data', i.e., the data 
containing no noise was not exactly zero in megawatts (MW) scale with respect to the superheater system. The corresponding difference
in Q (MW) is also provided in the excel spreadsheet with respect to the true data obtained by simulation of the first-principles model.
Although the specific formulation of the training and forward problems remain independent of the specfic types of constraints imposed, 
in this work, it is desired to see whether the ECNN models converge at the 'True Data' considered even when trained against noisy transient 
data. Therefore, the corresponding constraint equations are modified to arrive at the Delta_Q (MW) values calculated with respect to the 'True Data'.

## Color Code

The input and output variables have been color-coded in both excel spreadsheets attached for clarity --
  * The columns representing the variation in model inputs shown in 'light orange'
  * The columns representing the variation in corresponding model outputs shown in 'light green'
  * The columns representing additional parameters / thermo-physical properties required for imposition of
    energy constraints for the system (such as specific heats, etc.) shown in 'light purple'. While the specific
    heat capacities of the flue gas stream are assigned a constant value based on the Ideal Gas laws, the specific
    enthalpies of steam have been calculated using functions defined in the IAPWS R7-97 (MATLAB) / XSteam (Python)
    formulation for the properties of water and steam in terms of steam inlet / outlet temperature and pressure.

