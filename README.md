# Deploy inference endpoints using a custom spark 3.2.0 image

## Requirements

* Create an [Azure Machine Learning workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=python)
* Install [azureml-sdk](https://pypi.org/project/azureml-sdk/)

You can use this tutorial with Jupyter notebooks (from Azure ML) as well as Azure Databricks notebooks.

## Register an inference spark environment

We will use a custom Dockerfile combining the Prebuilt Azure ML environment [Minimal-Ubuntu-18](https://hub.docker.com/_/microsoft-azureml-inference) with Spark 3.2 image from [SynapseML](https://github.com/microsoft/SynapseML/blob/master/tools/docker/minimal/Dockerfile).

### Get the Azure ML Workspace

In this case we create a `get_workspace()` method that uses **azureml-sdk** and a [`Service Principal Authentication`](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication). To ensure we will connect to the Azure ML Workspace in a secure way, we will also use a [Key Vault](https://docs.microsoft.com/en-us/azure/key-vault/general/overview) to store our keys. Please see this [doc](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication) to check the different auth methods.

```python
import azureml
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import Environment

workspace_name = '<YOUR-WORKSPACE-NAME>'
resource_group = '<YOUR-RESOURCE-GROUP>'
subscription_id = '<YOUR-SUBSCRIPTION-ID>'

def get_workspace(workspace_name, resource_group, subscription_id):
  svc_pr = ServicePrincipalAuthentication(
      tenant_id = dbutils.secrets.get(scope = "azure-key-vault", key = "tenant-id"),
      service_principal_id = dbutils.secrets.get(scope = "azure-key-vault", key = "cliente-id-custom-role"),
      service_principal_password = dbutils.secrets.get(scope = "azure-key-vault", key = "cliente-secret-custom-role"))

  workspace = Workspace.get(name = workspace_name,
                            resource_group = resource_group,
                            subscription_id = subscription_id,
                            auth=svc_pr)
  
  return workspace

workspace = get_workspace(workspace_name, resource_group, subscription_id)
```

### Create the environment

With Azure ML we can register a [environment](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments#:~:text=By%20default%2C%20Azure%20ML%20will%20build%20a%20Conda,libraries%20that%20you%20installed%20on%20the%20base%20image.) to track and reproduce our projects' software dependencies as they evolve.

```python
from azureml.core.environment import Environment
from azureml.core.webservice import LocalWebservice
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig, Model

# BASE IMAGE from https://github.com/Azure/mmlspark/blob/master/tools/docker/minimal/Dockerfile 
dockerfile = """
FROM mcr.microsoft.com/azureml/minimal-ubuntu18.04-py37-cpu-inference:latest

USER root:root

ARG SPARK_VERSION=3.2.0
RUN mkdir -p /usr/share/man/man1 && \
    apt-get -qq update && apt-get \
    -qq upgrade && apt-get -qq -y install curl bzip2 && \
    curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && rm -rf /tmp/miniconda.sh && \
    conda install -y python=3 && \
    conda update conda && \
    apt-get install openjdk-8-jre -y && \
    conda install -c conda-forge pyspark=${SPARK_VERSION} && \
    apt-get -qq -y remove curl bzip2 && \
    apt-get -qq -y autoremove && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log && \
    conda clean --all --yes

RUN pip install mlflow pyspark pandas azureml-defaults

ENV PATH /opt/conda/bin:$PATH
ENV JAVA_HOME /usr/lib/jvm/java-1.8.0-openjdk-amd64
"""

# Create an environment to be able to customize our dependencies
my_spark_env = Environment('spark-env-custom')

# Now we can indicate we will use our custom Base Image
my_spark_env.docker.base_image = None
my_spark_env.docker.base_dockerfile = dockerfile
my_spark_env.python.user_managed_dependencies = True

# It's very important to use this parameter
my_spark_env.inferencing_stack_version='latest'
```

### Register the environment

Now that we have created the environment with all dependencies we need, we can simply register it to be able to use it when necessary.

`my_spark_env.register(workspace)`

## Deploy from an environment

To get this environment we can use `Environment.get` from azureml-sdk as well, so we can get the frozen environment to reuse it when it's being defined the **Inference Config** in the deployment process.

```python
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig

my_spark_env = Environment.get(name='spark-env-custom', workspace=workspace)

inference_config = InferenceConfig(entry_script="<YOUR-ENTRY-SCRIPT>", environment=my_spark_env)
```

## Finally deploy the model

Here we will use ACI (Azure Container Instances) to deploy our model, but feel free to use the environment to an AKS deployment as well.

```python
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice

service_name = 'api-model-dev'

aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, description="This is a spark serving example.")
 
service = Model.deploy(name=service_name, deployment_config=aci_config, models=[model_azure], inference_config=inference_config, workspace=workspace, overwrite=True)
service.wait_for_deployment(show_output=True)
```

Here the notebooks with the full example. Please feel free to try with your own spark model:

* [Train a native spark model](./notebooks/train-spark-classification-model.ipynb)
* [Register an inference spark environment](./notebooks/register-spark-environment.ipynb)
* [Deploy a spark model from an environment](./notebooks/deploy-from-environment.ipynb)
