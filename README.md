#  Project: Predict Flight Delays and Cancellations During the Holiday Season
Yuchen Xu, Mario Ma, Yiteng Tu, Yudi Wang.

## Description
The holiday season (November to January) is one of the busiest times for airlines, with frequent delays and cancellations. This project uses data from the U.S. Department of Transportation and National Weather Service to identify patterns in holiday travel disruptions and offer practical tips for passengers to minimize delays and cancellations. Additionally, a predictive model will be developed to estimate gate arrival times, helping travelers better plan their holiday journeys.

## Repository Structure

### 1. Code
- **match1.ipynb**:
  - Organizes initial flight data. Calculates the distance between latitude and longitude coordinates using the Haversine formula to match each airport with its corresponding weather station.
- **match_faileddownload.ipynb**:
  - Finds the second and third closest weather stations for airports where the initial data download failed.
- **airporttimezone.ipynb**:
  - Determines the timezone associated with each airport.
- **timezone_flight2.ipynb**:
  - Standardizes the four key times in the flight data (CRSDepTime, CRSArrTime, DepTime, ArrTime) to Central Standard Time (CST).

### 2. Data
- **Airport situation.csv**: Contains the raw data used for this project.
- **ghcnh-station-list.csv.csv**: Contains the raw data used for this project.
- **Airport situation unique.csv**: Contains the raw data used for this project.

### 3. Image
- Contains various images and plots generated during the data analysis and modeling stages.

### 4. Summary
- A document summarizing the key steps in data cleaning, model building, and model evaluation from the project.

### 5. Slides
- Presentation slides summarizing the project, including the datacleaning, methodology, and results.

## Shiny Link
The Shiny app allows users to interactively input measurements and predict body fat percentage based on the trained model. You can access the live app here:
- [Shiny App Link](https://mario2747.shinyapps.io/)
