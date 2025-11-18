@echo off
echo Deleting old model files to retrain with improved thresholds...
del heart_disease_model.pkl 2>nul
del scaler.pkl 2>nul
del disease_type_model.pkl 2>nul
echo Old models deleted!
echo.
echo Starting server - the model will retrain automatically...
python app.py
pause

