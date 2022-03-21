# Amazon SageMaker Safe Deployment Pipeline for AICovidVN Solution (aishield)

###  Approximate Times (Expectation):

The following is a list of approximate running times for the pipeline:

* Full Pipeline: 35 minutes
* Start Build: 2 minutes
* Model Training and Baseline: 5 minutes
* Launch Dev Endpoint: 10 minutes
* Launch Prod Endpoint: 15 minutes
* Monitoring Schedule: runs on the hour

## Directory & file structures (Expectation)

This project is written in Python, and design to be customized for your own model and API.

```
.
├── api
│   ├── __init__.py
│   ├── app.py
│   ├── post_traffic_hook.py
│   └── pre_traffic_hook.py
├── assets
│   ├── deploy-model-dev.yml
│   ├── deploy-model-prod.yml
│   ├── suggest-baseline.yml
│   └── training-job.yml
├── custom_resource
|   ├── __init__.py
|   ├── sagemaker_monitoring_schedule.py
|   ├── sagemaker_suggest_baseline.py
|   ├── sagemaker_training_job.py
│   └── sagemaker-custom-resource.yml
├── model
│   ├── buildspec.yml
│   ├── dashboard.json
│   ├── requirements.txt
│   └── run_pipeline.py
├── notebook
│   ├── dashboard.json
|   ├── workflow.ipynb
│   └── mlops.ipynb
├── scripts
|   ├── build.sh
|   ├── lint.sh
|   └── set_kernelspec.py
├── pipeline.yml
└── studio.yml
```

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
