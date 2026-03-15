# Deep learning applications for Auroral maps
<img src="https://github.com/jah-26603/GUVI-Auroral-Deep-Learning-Model/blob/main/images/raw_guvi_1_10_hour_forecast.png" alt="Aurora" width="400">   <img src="https://github.com/jah-26603/GUVI-Auroral-Deep-Learning-Model/blob/main/images/raw_guvi_1.png" alt="Aurora" width="400">

The current implementation is a simple pipeline that downalods all of the necsessary data, and trains a GUVI based auroral preciptation model. There are multiple modes of this training where the inputs to the model are 
strictly solar wind data for 4 hours or 72 hour context windows, or solar wind data (4 hours) + solar activity + geomagnetic activity + other indices. The model predictions shown are using the largest context window, with 
only solar wind data being used. 

<br>
<br>
The next step is to apply a diffusion based method to generate global assimilative maps instead of the mean likely state given a set of solar wind data. 



   
