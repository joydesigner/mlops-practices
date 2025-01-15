[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](LICENSE)

# DevOps for Machine Learning | MLOps

This repository offers practical guidance on implementing MLOps workflows using [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/) and [Azure DevOps](https://docs.microsoft.com/en-us/azure/devops/?view=azure-devops&viewFallbackFrom=vsts).

![ML Loop](./architecture/ml-loop.PNG)

## Overview

Machine Learning Operations ([MLOps](https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment)) applies DevOps principles to the machine learning lifecycle, enhancing workflow efficiency and model reliability.

This repository provides code samples and guidelines to configure MLOps workflows on Azure, as illustrated below:

![Workflow Diagram](./architecture/flow.PNG)

## Features

Azure Machine Learning offers several MLOps capabilities:

- **Machine Learning Pipelines**: Define repeatable and reusable steps for data preparation, training, and scoring processes.

- **Reusable Software Environments**: Create consistent environments for training and deploying models.

- **Model Management**: Register, package, and deploy models with associated metadata from any location.

- **Governance**: Capture comprehensive data throughout the ML lifecycle, including publication history, change reasons, and deployment records.

- **Alerts and Notifications**: Receive updates on key events such as experiment completion, model registration, deployment, and data drift detection.

- **Monitoring**: Oversee ML applications for operational and performance issues, comparing model inputs between training and inference, and exploring model-specific metrics.

- **Automation**: Utilize Azure Machine Learning and Azure Pipelines to automate the end-to-end ML lifecycle, facilitating frequent model updates and continuous deployment.

![ML Lifecycle](./architecture/ml-lifecycle.png)

## Repository Structure

This repository includes the following key files:

- **`job.yml`**: Defines the Azure Machine Learning job configuration, specifying the training script, inputs, environment, compute resources, and experiment details.

- **`train_insurance.runconfig`**: Specifies the runtime configuration for the training job, including environment settings, data references, and Docker image details.

- **`parameters.json`**: Contains hyperparameters and training configurations used by the training script to fine-tune the machine learning model.

- **`train_aml.py`**: The main training script that orchestrates data loading, model training, evaluation, and registration within Azure Machine Learning.

- **`train_test.py`**: Includes unit tests for the training functions to ensure code reliability and correctness.

- **`train.py`**: Contains core functions for data splitting, model training, and evaluation metrics computation.

- **`conda_dependencies.yml`**: Lists the dependencies and packages required for the project, facilitating environment setup and reproducibility.

## Getting Started

To explore and implement these MLOps practices:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/joydesigner/mlops-practices.git

2. **Set Up Azure Services**:
- Ensure you have access to Azure Machine Learning and Azure DevOps.

3. **Configure the Environment**:
- Create a new Conda environment using the provided conda_dependencies.yml file:

   ```
   conda env create -f conda_dependencies.yml
   conda activate project_environment
   ```

4. **Run the Training Script**:

   ```
   python train_aml.py --data_file_path <path_to_data> --model_name <model_name>     --dataset_name <dataset_name>
   ```

5. **Monitor the Experiment**:
- Use Azure Machine Learning Studio to monitor the experiment, check logs, and evaluate model performance.
