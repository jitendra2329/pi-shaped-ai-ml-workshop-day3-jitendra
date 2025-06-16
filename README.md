
---

## Features & Methods

### Data Cleaning
- Removes rows with missing values in key delay columns.
- Filters out airports with fewer than 100 data points.

### Feature Engineering

1. **Time-Based Features**
   - Extracts:
     - `quarter` from `month`
     - `season` based on the month number

2. **Categorical Encoding**
   - Applies `Label Encoding` to:
     - `airport`
     - `carrier`
   - Applies `One-Hot Encoding` to:
     - `season`

3. **Interaction Features**
   - Combines `carrier` and `airport` into a single string (e.g., `"AA_JFK"`) and encodes it.

4. **Aggregated Features**
   - Calculates:
     - `avg_carrier_delay` per carrier
     - `total_flights_by_airport` per airport

---

## Model Training

- **Model:** Random Forest Classifier
- **Target:** Airport label (encoded)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score

---

## How to Run

```bash
# Install required libraries
pip install -r requirements.txt

# Run the pipeline
python main.py
