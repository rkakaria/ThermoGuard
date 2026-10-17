# ThermoGuard

Overheating is one of the most common causes of hardware failure, slowdowns, and data loss in laptops. Users usually notice only after performance drops, fans get loud, or the system shuts down. Tools may show raw sensor numbers that most users cannot interpret, or rely on fixed temperature thresholds that ignore context (a temperature that is normal under heavy CPU load may be abnormal at idle, especially if the fan is running slowly). We need a system that reads machine health readings, recognizes overheating (and related hardware problems) early, and tells the user clearly what is happening and what to do. 

The goal of this project is to build a system that takes laptop health readings (CPU usage, RAM usage, motherboard temperature, voltage, disk usage, fan speed) and classifies the machine's condition using a trained ML model. The primary goal is reliable overheating detection; the system will also flag the other problem types present in the dataset (power issue, memory leak, disk failure). The system will: 
- Classify each reading as one of: No Problem, Overheating, Power Issue, Memory Leak, or Disk Failure. 
- Explain each alert in plain language with a suggested action. 
- Show history and trends so users can see patterns over time. 
- Improve over time using user feedback on false alarms and missed alerts. 
