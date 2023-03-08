## Submission Ledger:

### 1. 
Baseline models --> RandomForest, XGBoost, LightGBM, CatBopost
Set of Features --> 'distance', 'haversine', 'duration', 'passenger_count', 'pickup_day','Monday', 'Tuesday', 'Wednesday', 'Thursday',  'Friday', 'Saturday','weekend', 'pickup_hour', 'rush_hour', 'overnight', 'pickup_LGA', 'dropoff_LGA', 'pickup_JFK', 'dropoff_JFK', 'pickup_EWR', 'dropoff_EWR', 'airport', 'change_borough'

### 2. 
Second round of models with optimized hyper-parameters --> RandomForest, XGBoost, LightGBM, Ensemble method (stacking)
Set of Features --> 'distance', 'haversine', 'duration', 'passenger_count', 'pickup_day','Monday', 'Tuesday', 'Wednesday', 'Thursday',  'Friday', 'Saturday','weekend', 'pickup_hour', 'rush_hour', 'overnight', 'pickup_LGA', 'dropoff_LGA', 'pickup_JFK', 'dropoff_JFK', 'pickup_EWR', 'dropoff_EWR', 'airport', 'change_borough'
Main Changes --> Using optimized HPs instead of standard. 

### 3. 
Third round of models with optimized hyper-parameters --> RandomForest, XGBoost, LightGBM, Ensemble method (stacking)
Set of Features --> 'distance', 'haversine', 'duration', 'passenger_count', 'pickup_day','Monday', 'Tuesday', 'Wednesday', 'Thursday',  'Friday', 'Saturday','weekend', 'pickup_hour', 'rush_hour', 'overnight', 'pickup_LGA', 'dropoff_LGA', 'pickup_JFK', 'dropoff_JFK', 'pickup_EWR', 'dropoff_EWR', 'airport', 'change_borough', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'
Main Changes --> Added coordinates to input features.