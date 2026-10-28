# Industrial Equipment Fault Prediction

Industrial rotating equipment is subject to faults, which can be signaled by overheating, acoustic signals, or excessive vibration. Users may not notice measurements out of the ordinary until the equipment actually fails. Tools may show raw sensor numbers that most users cannot interpret. We need a system that reads machine health readings, recognizes overhearing and hardware problems early, and tells the user clearly what is happening and what to do. 

The goal of this project is to build a system that takes rotating equipment health readings (vibration in the X, Y, and Z axes; acoustic sound levels; and temperature) and classifies the machine's condition using a trained ML model. The primary goal is reliable fault detection. The system will: 
- Classify each reading as one of the following: `Normal`, `Bearing Fault`, `Imbalance`, or `Overheating`. 
- Explain each alert in plain language with a suggested action. 
- Show history and trends so users can see patterns over time. 

We utilize the [Rotating Equipment Multi Sensor Fault Dataset](https://www.kaggle.com/datasets/zoya77/rotating-equipment-multi-sensor-fault-dataset) from Kaggle user zoya77 to train our ML model. 

##  How to Run
```
git clone https://github.com/rkakaria/industrial-equipment-fault-prediction.git
python app.py
```

## Dependencies
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [pandas](https://pandas.pydata.org/)
- [gradio](https://github.com/gradio-app/gradio)