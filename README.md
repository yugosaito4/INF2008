# INF2008
Machine Learning Project by P5-17

## Members
1. Dennis Tan Ming Zhen (2301789)
2. Saito Yugo (2301793)
3. Lim Miang How Matthew (2301820)
4. Tan Jia Xin Vanessa (2302059)
5. Aisyah Nur Imanah Binti Samsuri (2301921)
6. Wong Qing Liang (2302163)


> [!IMPORTANT]
> 1. The features are already extracted in the 'Features' folder
> 2. Re-Running  `extraction.py` will override the exisiting feature parquet file. Change the file path for *CK+/FER2013* set
> 3. Latest Logging details of the model training which includes the confusion matrix is in the 'Logging' folder
> 4. Weighted SVM model is tuned for FER2013 set while SVM is for CK+ Dataset 


## User Manual
**Environment Setup**
1.Clone the GitHub repository from https://github.com/yugosaito4/INF2008
2.Prerequisites:
⋅⋅*Python 3.8+
⋅⋅*pipenv installed
3.Install all required packages by navigating to the root of the project directory and running the command pipenv install in the terminal.

**Data Processing**
1.Dataset is under the ‘images’ folder.
2.The features are already extracted in the 'Features' folder
3.Re-running extraction.py will override the existing feature parquet file. Make sure to update the file path inside extraction.py if you wish to switch between CK+ and FER2013 datasets.

**Training / Testing**
1.For CK+ Dataset (using SVM model): Run SVM.py
2.For CK+ Dataset (using RandomForest model): Run RandomForest(Controlled).py
3.For FER2013 Dataset (using SVM model): Run weighted_SVM.py
4.For FER2013 Dataset (using RandomForest model): Run RandomForest(Complex).py

**Results**
1.Evaluation results (e.g., accuracy, confusion matrix) will be printed in the terminal.
