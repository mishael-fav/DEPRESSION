
# Depression Analysis Among Students 📊🧠

## Project Overview

**Depression** is a common but serious mental health condition that affects how individuals think, feel, and function in daily life. It can lead to emotional and physical problems, and in severe cases, even suicidal thoughts.

Among various social classes and demographics, students—especially in secondary and tertiary institutions—are increasingly vulnerable to depression due to:

- Academic pressure  
- Social expectations  
- Poor coping mechanisms  
- Inadequate support systems  

**Goal:**  
Understanding the patterns and predictors of depression among students is critical. It helps in:

- Designing timely interventions  
- Shaping school mental health policies  
- Promoting emotional well-being  

This analysis aims to identify key risk factors and target support efforts toward the most affected groups, ultimately fostering a healthier learning environment.

---

## Dataset

- **Source:** [Kaggle - Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
- **Attributes:**
  - **Age:** 18 - 59 years (mean ~26 years)
  - **CGPA:** Max value 10, average 7.66
  - Majority of respondents are **college/university students** or early professionals.

---

## Tools & Libraries Used

- **Python 3**
- **Pandas** - Data manipulation and analysis  
- **NumPy** - Numerical computing  
- **Matplotlib** - Data visualization  
- **Seaborn** - Advanced visualization  
- **GeoPandas** - Geospatial data handling  

---

## Process

1. **Data Loading**
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import geopandas as gpd
    
    Depression_data = pd.read_csv('/content/drive/MyDrive/student_depression_dataset.csv')
    ```

2. **Data Exploration**
    - Display data structure:
        ```python
        Depression_data.head()
        Depression_data.info()
        Depression_data.duplicated().sum()
        ```
    - Check for missing values and data types.
    - Understand the distribution of key variables (Age, CGPA, Depression indicators).

3. **Feature Engineering & Cleaning**
    - Remove duplicates.
    - Handle missing values if applicable.
    - Prepare variables for visualization and modeling.

4. **Visualizations**
    - Age distribution of students.
    - Relationship between CGPA and depression level.
    - Impact of gender and profession on depression risk.

5. **Modeling & Insights**
    - Perform correlation analysis.
    - Highlight patterns/trends from the data.
    - Recommend interventions based on findings.

---

## Key Insights

✅ Majority of the respondents fall within the early 20s age group—indicating this cohort is particularly vulnerable.  
✅ CGPA and academic stress show relationships with depression levels.  
✅ Gender and profession also contribute to differing depression risks.  
✅ Targeted mental health support is needed for **students** at risk.

---

## Next Steps

- Expand analysis with more features (e.g., social activity, lifestyle habits).
- Explore predictive modeling (Logistic Regression, Random Forest) for depression likelihood.
- Integrate results with **school support programs**.

---

## Conclusion

This project is a small but important step toward using **data science for social good**.  
Understanding student depression patterns allows institutions to implement better **preventive** and **responsive** measures.

---
