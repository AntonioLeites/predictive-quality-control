# 🏭 Predictive Quality Control in Manufacturing
## A Logistic Regression Proof-of-Concept

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Companion repository for LinkedIn article:** *From €50K/Month Losses to Real-Time Quality Control*

---

## 🎯 Project Overview

This proof-of-concept demonstrates how **logistic regression** can enable real-time predictive quality control in manufacturing environments. While simple compared to deep learning approaches, logistic regression offers critical advantages:

- ✅ **Interpretable:** Engineers understand why each decision is made
- ✅ **Fast:** <1ms predictions suitable for high-speed production lines
- ✅ **Transparent:** Coefficients reveal root causes of quality issues
- ✅ **Production-ready:** Easy to deploy and maintain

### ⚠️ Important Disclaimer

**This is a proof-of-concept using synthetic data.** All sensor readings, production parameters, and defect outcomes are artificially generated to model realistic manufacturing scenarios. The methodology is sound and production-ready; the specific numbers demonstrate potential value but are not from actual production systems.

---

## 📊 Dataset Specifications

### Synthetic Data Generation

We created a realistic dataset simulating an **electronics component manufacturing line**:

```python
Total samples: 2,000 production parts
Training set: 1,500 samples (75%)
Test set: 500 samples (25%)
Defect rate: 5% (100 defective parts in full dataset)
```

### Features (12 parameters)

**Machine Parameters:**
- `oven_temperature_c`: Oven temperature (°C) - Normal(220, 15)
- `molding_pressure_bar`: Molding pressure (bar) - Normal(150, 20)
- `line_speed_mpm`: Line speed (meters/min) - Normal(45, 5)
- `ambient_humidity_pct`: Ambient humidity (%) - Normal(45, 10)

**Material Parameters:**
- `material_thickness_mm`: Material thickness (mm) - Normal(2.5, 0.3)
- `material_strength_mpa`: Material strength (MPa) - Normal(350, 40)

**Operational Parameters:**
- `cycle_time_sec`: Cycle time (seconds) - Normal(12, 2)
- `machine_vibration_hz`: Machine vibration (Hz) - Uniform(0.5, 3.5)
- `tool_age_hours`: Tool age (hours) - Uniform(0, 500)

**Context:**
- `shift`: Shift (1=Morning, 2=Afternoon, 3=Night)
- `operator_experience_years`: Operator experience (years) - Uniform(1, 20)
- `days_since_maintenance`: Days since last maintenance - Uniform(1, 30)

### Defect Generation Logic

Defects are generated based on a **weighted risk score** to simulate realistic manufacturing behavior:

```python
defect_score = (
    0.15 × |temperature_deviation_from_220°C| +
    0.10 × |pressure_deviation_from_150bar| +
    0.12 × vibration_level +
    0.08 × (tool_age / 100) +
    0.05 × days_since_maintenance +
    -0.03 × operator_experience +
    random_noise
)

# Top 5% of risk scores → labeled as defective
```

This creates a **realistic 5% defect rate** with clear causal relationships that the model can learn.

---

## 🔬 Model Performance

- **Model:** Logistic Regression

- **ROC-AUC:** ~0.63 (illustrative with synthetic data)

### Threshold Flexibility
The REST API supports a custom threshold for classification (default = 0.5):

```
{"defect_probability": 0.06, "predicted_defect": 0, "threshold_used": 0.5}
{"defect_probability": 0.06, "predicted_defect": 0, "threshold_used": 0.3}

```
- Lower threshold → higher recall, more false positives

- Higher threshold → fewer false positives, lower recall


---

## 💰 Business Impact (Illustrative)



- Production volume: 500,000 parts/year
- Defect rate: 5%
- Cost per defect reaching customer: €500
- Cost per false positive inspection: €5

**Net annual savings:** ~€1.6M (synthetic scenario)




## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/AntonioLeites/predictive-quality-control.git
cd predictive-quality-control

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
You have **two alternatives** to explore this PoC:

- Run the Jupyter Notebook (Full Analysis)
- Step-by-Step Execution
- 

### Run the Analysis

```bash
# Open Jupyter Notebook
jupyter notebook

# Open: notebooks/predictive_quality_control.ipynb
# Run all cells (Ctrl+Enter)
```
### Expected Output

The notebook will:
1. Generate synthetic dataset (2,000 samples)
2. Train logistic regression model
3. Display performance metrics
4. Show 7 visualizations:
   - Confusion matrix
   - ROC curve
   - Feature importance
   - Temperature vs defects
   - Tool age analysis
   - Quality by shift
   - Real-time sensor trends

Full analysis report is saved as:
docs/Quality_Control_System_FullAnalysis.png

![Quality Control System - Full Analysis](Quatlity_Control_System_FullAnalysys.png)

### Step-by-Step Execution
1.  Generate synthetic data
```bash
   python src/data_generation.py
```

2. Train the logistic regression model
```bash
   python src/model_training.py
```
3. Run the REST API
```bash
   uvicorn src.prediction_api:app --reload
```
4. Make predictions via API
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict?threshold=0.5" -H "Content-Type: application/json" -d '{\
   "oven_temperature_c": 240, \
   "molding_pressure_bar": 160, \
   "line_speed_mpm": 46, \
   "ambient_humidity_pct": 40,  \
   "material_thickness_mm": 2.5,  \
   "material_strength_mpa": 355,  \
   "cycle_time_sec": 12,  \
   "machine_vibration_hz": 1.5,  \
   "tool_age_hours": 420,  \
   "shift": 2,  \
   "operator_experience_years": 3,  \
   "days_since_maintenance": 15  \
   }'

## 🐳 Run with Docker

You can containerize and serve the Predictive Quality Control API using Docker or Docker Compose.

### Option 1: Run with Docker

Make sure you’re in the project root (where the deployment folder exists), then build and run the container:

```bash
# Build the image
docker build -f deployment/Dockerfile -t predictive-quality-control .

# Run the container
docker run -d -p 8000:8000 --name predictive-quality-control predictive-quality-control
```
Once running, open the API documentation in your browser:

👉 http://127.0.0.1:8000/docs

---
### Option 2: Run with Docker Compose (recommended)

The project also includes a docker-compose.yml file that makes it easier to build and launch the API with the trained model automatically mounted.

```bash
   docker-compose up --build
```

This will:
- Build the image using deployment/Dockerfile
- Expose port 8000
- Mount the models/ folder to /app/models in the container
- Start the FastAPI server

Stop the container anytime with:
```bash
   docker-compose down
```
### Test the API

Once the container is running, you can send a POST request with sample sensor data, e.g.:
```bash
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.3" \
   -H "Content-Type: application/json" \
   -d '{
      "oven_temperature_c": 230,
      "molding_pressure_bar": 160,
      "line_speed_mpm": 47,
      "ambient_humidity_pct": 50,
      "material_thickness_mm": 2.6,
      "material_strength_mpa": 355,
      "cycle_time_sec": 13,
      "machine_vibration_hz": 2.1,
      "tool_age_hours": 220,
      "shift": 2,
      "operator_experience_years": 8,
      "days_since_maintenance": 10
   }'

```
Expected response:
```bash
{
  "defect_probability": 0.27,
  "predicted_defect": 0,
  "threshold_used": 0.3
}

```
---
## ☁️ Integration with SAP BTP and S/4HANA

Once the Predictive Quality Control API is containerized, it can be deployed and consumed in your SAP BTP landscape to extend S/4HANA manufacturing or quality processes.

## 1️⃣ Deploy to SAP BTP

Once the Predictive Quality Control API is containerized, it can be deployed into **SAP BTP** to be consumed by applications, **SAP Joule skills**, and **S/4HANA** quality processes.

You can deploy this project in two ways:

### **Option 1 — Deploy to SAP BTP Kyma Runtime**

Push the container image to your registry (Docker Hub, GitHub Container Registry, or SAP GAR):

```bash
docker tag predictive-quality-control <your_registry>/predictive-quality-control:latest
docker push <your_registry>/predictive-quality-control:latest
```
### **Option 2 — Deploy to SAP AI Core** (recommended for scalable model serving)
This deployment model is used when the inference service should run in a managed, scalable compute environment, and optionally when retraining is executed in SAP AI Core pipelines.

#### **0. Prepare Artifacts**

| Component / File | Purpose |
|---------|---------|
| `Dockerfile.train`|Used when training will run in AI Core |
| `Dockerfile.serve` | Used to serve inference as an API |
| `training-template.yaml`  | Contains argo file for defining Scenario |
| `serving-template.yaml`  | Contains argo file for defining Scenario |


If the registry is private, create an AI Core registry secret for image pulling.

#### **1. Add a Git Repository**
You can use your own git repository to version control your SAP AI Core templates. The GitOps onboarding to
SAP AI Core instances involves setting up your git repository and synchronizing your content.
You will need to generate a personal access token for your git repository.
```bash
# Include the training template (WorkflowTemplate) and the serving template (ServingTemplate)
deployment/training-template.yaml
deployment/serving-template.yaml
```
#### **2. Create an Application**
After registering your Git repository, create an application to sync the templates in your repository.
After the GitOps setup is completed, the templates in your git repository are automatically synced to SAP AI
Core. 

This will create your scenario. 
A scenario is an implementation of a specific AI use case within a user's tenant. It consists of a pre-defined set
of AI capabilities in the form of executables and templates.
#### **2. Create Training Configuration**

A configuration is a collection of parameters, artifact references (such as datasets or models), and
environment settings that are used to instantiate and run an execution or deployment of an executable or
template.


#### **3. Create Execution for Training**
Workflow templates are built on the Argo Workflows engine and are defined as WorkflowTemplates.
These are your cluster's workflow definitions.

#### **4. Create the Serving Configuration**
Deploy а trained machine learning model as a Web service to serve inference
requests of trained models with high performance.
The serving templates are used to create model servers. When a model server is up and running, it processes
incoming inference requests and returns the results from the AI learning model. Serving templates define how
a model is to be deployed.



#### **5. Deploy the Model Serving Endpoint**
The duration of a deployment can be limited using the ttl parameter. It takes an integer for quantity, and
a single letter to specify units of time. Only minutes (m), hours (h) and days (d), are supported, and values
must be natural numbers. For example, "ttl": "5h" gives the deployment a duration of 5 hours. 4.5h and
4h30m are not valid inputs. If no value is passed, the duration of the deployment if indefinite. Once the duration
expires, the deployment is stopped and deleted..

#### **6. Retrieve the Public Inference Endpoint**

Use the URL from your model deployment to access the results of your model.

---

## 🏗️ Enterprise Architecture: SAP Integration

This POC becomes production-ready when integrated with SAP systems:
```
┌─────────────────────────────────────────────────────────────┐
│                    SAP S/4HANA Manufacturing                │
│  ┌────────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ Production     │  │   Quality   │  │   Equipment     │   │
│  │ Orders (PP)    │  │   Mgmt (QM) │  │   Master Data   │   │
│  └────────────────┘  └─────────────┘  └─────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │ OData APIs / CDS Views
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               SAP Plant Connectivity (PCo)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Real-time Sensor Data (OPC-UA / MQTT)               │   │
│  │  • Temperature  • Pressure  • Vibration  • Tool Age  │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    SAP BTP AI Core                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Logistic Regression Model (REST API)                │   │
│  │  Input: Sensor readings → Output: Defect probability │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   SAP Joule Studio                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Skill 1:     │  │ Skill 2:     │  │ Skill 3:         │   │
│  │ Get Sensors  │→ │ Predict Risk │→ │ Root Cause       │   │
│  └──────────────┘  └──────────────┘  │ Analysis         │   │
│                                       └──────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   SAP Joule (User Interface)                │
│                                                             │
│  User: "Why is Line 2 showing high defect risk?"            │
│                                                             │
│  Joule: "Line 2 has 68% defect risk because:                │
│         • Oven temp at 245°C (target: 220°C)                │
│         • Tool age 420h (recommend change at 400h)          │
│         Recommended: Replace tool + recalibrate oven"       │
└─────────────────────────────────────────────────────────────┘
```


### Roadmap

#### **Phase 1: Proof of Concept**

1. **Deploy ML Model to BTP AI Core** 
   - Package model as Docker container
   - Deploy to AI Core with auto-scaling
   - Expose REST endpoint

#### **Phase 2: Next Steps**

2. **Create Joule Skills** 
   - `getSensorData(lineId)` → Plant Connectivity
   - `predictDefectRisk(sensors)` → AI Core
   - `getProductionContext(lineId)` → S/4HANA PP/QM
   - `analyzeRootCause(lineId)` → Orchestrates above

3. **Configure Plant Connectivity** 
   - Install PCo agents on production line
   - Map SCADA tags to SAP structure
   - Enable real-time streaming

4. **Enable Automated Actions** 
   - Risk > 70% → Create QM notification
   - Risk > 85% → Trigger maintenance workflow
   - Daily summary → Email to quality manager

---

## 📁 Repository Structure

```
predictive-quality-control/
│
├── notebooks/
│   └── predictive_quality_control.ipynb    # Main analysis
│
├── src/
│   ├── data_generation.py                  # Synthetic data creation
│   ├── model_training.py                   # Model training pipeline
│   ├── prediction_api.py                   # REST API for deployment
│   └── utils.py                            # Helper functions
│
├── data/
│   ├── synthetic_production_data.csv       # Generated dataset
│   └── data_dictionary.json                # Feature descriptions
│
├── models/
│   └── logistic_regression_v1.pkl          # Trained model
│
├── deployment/
│   ├── Dockerfile                          # Container for BTP
│   ├── requirements.txt                    # Python dependencies
│   └── joule_skills/                       # Joule Studio skill definitions
│       ├── get_sensor_data.yaml
│       ├── predict_defect_risk.yaml
│       └── analyze_root_cause.yaml
│
├── docs/
│   ├── technical_specifications.md         # Detailed specs
│   ├── cost_benefit_analysis.xlsx          # ROI calculations
│   └── sap_integration_guide.md            # Step-by-step SAP setup
│
├── tests/
│   ├── test_model.py                       # Unit tests
│   └── test_api.py                         # API tests
│
├── README.md                               # This file
├── LICENSE                                 # MIT License
└── .gitignore
```

---

## 🛠️ Technical Stack

- **Python 3.8+**
- **scikit-learn 1.3+** (LogisticRegression, StandardScaler, GridSearchCV)
- **pandas 2.0+** (Data manipulation)
- **numpy 1.24+** (Numerical operations)
- **matplotlib 3.7+** (Visualizations)
- **Chart.js 4.4** (Dashboard visualizations)

**Deployment:**
- **SAP BTP AI Core** (Model hosting)
- **Flask/FastAPI** (REST API)
- **Docker** (Containerization)

---

## 🤝 Contributing

This is a proof-of-concept for educational purposes. If you'd like to:
- Extend to multi-class classification (defect types)
- Add time-series forecasting for preventive maintenance
- Implement with real production data
- Integrate with other MES/SCADA systems

Please open an issue or submit a pull request!

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📬 Contact & Discussion

**LinkedIn Article:** [Link to your LinkedIn post]

**Author:** Antonio Leites  
**LinkedIn:** https://www.linkedin.com/in/antonioleites/  
**Email:** antonio.leites-lopez@sulzer.de

---

## 🙏 Acknowledgments

- Dataset generation inspired by real manufacturing best practices
- Cost assumptions based on industry benchmarks from [cite sources if you have them]
- SAP integration architecture follows SAP BTP reference patterns

---

## ⭐ If this helped you...

Please give this repository a star ⭐ and share your experience implementing predictive quality control in your manufacturing environment!

**Questions?** Open an issue or reach out on LinkedIn.

---

**Last Updated:** October 2025  
**Version:** 1.0.0