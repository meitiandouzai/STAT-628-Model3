#  Project: Predict Flight Delays and Cancellations During the Holiday Season
Yuchen Xu, Mario Ma, Yiteng Tu, Yudi Wang.

## Description
The holiday season (November to January) is one of the busiest times for airlines, with frequent delays and cancellations. This project uses data from the U.S. Department of Transportation and National Weather Service to identify patterns in holiday travel disruptions and offer practical tips for passengers to minimize delays and cancellations. Additionally, a predictive model will be developed to estimate gate arrival times, helping travelers better plan their holiday journeys.

## Repository Structure

### 1. Code
- **match1.ipynb**:
  - Organizes raw airport and weather station data.
  - Calculates the distance between latitude and longitude coordinates using the Haversine formula to match each airport with its corresponding weather station.
- **match_faileddownload.ipynb**:
  - Finds the second and third closest weather stations for airports where the initial data download failed.
- **airporttimezone.ipynb**:
  - Determines the timezone associated with each airport.
- **timezone_flight2.ipynb**:
  - Organizes raw flight data.
  - Standardizes the four key times in the flight data (CRSDepTime, CRSArrTime, DepTime, ArrTime) to Central Standard Time (CST).
- **Merge1.ipynb**:
  - Select weather data for the holiday season (November-January).
  - Combine all flight data into a csv file.
- **Merge_final.ipynb**:
  - Merge the flight dataset and the weather dataset based on time converted to the same timezone. The merging method is as follows: for each flight sample, append the hourly weather data for both the departure and destination locations at the scheduled departure time to that row in the flight dataset.
- **model-cancelled-reg-lgb.ipynb** & **model_py_cancelled_randomforest.ipynb** & **model_cancelled_reg_final.ipynb**:
  - To predict flight cancellations, train logistic regression, LightGBM, and random forest models. For each model, we test whether to use SMOTE oversampling and whether to apply one-hot encoding. **model_cancelled_reg_final.ipynb** is the final version we selected.
- **model_cancelled_delay_*.ipynb** :
  - To predict flight arrival delay, train regression, LightGBM, random forest and neural network models.
- **finaldelaymodel.ipynb** :
  - The final model to predict flight arrival delay and interpret of the model. 
- **tips on delay.ipynb** :
  - The code for getting some tips on our delay model. 

### 2. Data
- **ghcnh-station-list.csv**: Contains raw data on weather station information.
- **T_MASTER_CORD.csv**: Contains raw airport data.
- **flight_holidayseason.zip**:[Link](https://drive.google.com/drive/folders/1v58ex2g1cIhyhanGa5GJoaqEuNIUv4dI?dmr=1&ec=wgc-drive-hero-goto) Contains raw filght data.
- **weather.zip**:[Link](https://drive.google.com/drive/folders/1v58ex2g1cIhyhanGa5GJoaqEuNIUv4dI?dmr=1&ec=wgc-drive-hero-goto) Contains raw weather data.
- **flight_processed.zip**:[Link](https://drive.google.com/drive/folders/1v58ex2g1cIhyhanGa5GJoaqEuNIUv4dI?dmr=1&ec=wgc-drive-hero-goto) Contains flight data with converted time zones.
- **Filtered_Airport_US_Unique.csv**: Contains airports with their latitude and longitude.
- **Airport_with_Nearest_Station.csv**: Contains airport latitude and longitude along with the nearest weather station and its distance..
- **failed_downloads.csv**: Contains airports where the initial data download failed.
- **Falied_Airport_with_Nearest_Station.csv**: Contains faildownload airport latitude and longitude along with the distances to the three nearest weather stations.
- **final_data.zip**:[Link](https://drive.google.com/drive/folders/1v58ex2g1cIhyhanGa5GJoaqEuNIUv4dI?dmr=1&ec=wgc-drive-hero-goto) Contains final data to fit the models.

### 3. Image
- Contains various images and plots generated during the data analysis and modeling stages.

### 4. Summary
- A document summarizing the key steps in data cleaning, model building, and model evaluation from the project.

### 5. Slides
- Presentation slides summarizing the project, including the datacleaning, methodology, and results.

### 6.Shiny
- Shiny code can be found in the folder called shiny. Two models are in the same folder.

## Shiny Link
The Shiny app allows users to interactively . You can access the live app here:
- [Shiny App Link](https://mario2747.shinyapps.io/flightpredict/)
