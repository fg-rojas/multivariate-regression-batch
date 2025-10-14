# multivariate-regression-batch-api

This repository shows an implementation in batch for a multivariate regression model already trained in model/model.pkl

To generate predictions using this model please follow the instructions:

## Requirements
- After cloning the repo, create a virtual environment using: 
```py -3.11 -m venv .venv```

Activate the environment
```.\.venv\Scripts\Activate.ps1```

Then install required libraries:
``` python -m pip install -r requirements.txt```

now you are ready to run the scorer_csv from console by using:
```python src/score_csv.py --model model/model.pkl --manifest model/manifest.json --input_csv data/blind_test_data.csv```


## Docker
You can also use docker to generate a new prediction csv file, 

first you need to build the docker image: 

```docker build -t mlt-regression-scorer:dev -f docker/Dockerfile .```

Then, you can generate the predictions file using the next command:

```docker run --rm  -v "${PWD}\data:/app/data"  -v "${PWD}\model:/app/model"  -v "${PWD}\out:/app/out"  mlt-regression-scorer:dev  --model /app/model/model.pkl  --manifest /app/model/manifest.json  --input_csv /app/data/blind_test_data.csv  --output_csv /app/out/out.csv```

### Test
This repo comes with a test to validate that the scorer_csv is behaving correctly when a dataframe with missing features is send, to run the test use:

```python -m pytest -q```