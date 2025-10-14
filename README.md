# multivariate-regression-batch-api

This repository shows an implementation in batch for a multivariate regression model already trained and persisted in model/model.pkl

To generate predictions using this model please follow the instructions:

## Requirements
- After cloning the repo, create a virtual environment using: 
```py -3.11 -m venv .venv```

- Activate the environment
```.\.venv\Scripts\Activate.ps1```

- Install required libraries:
``` python -m pip install -r requirements.txt```

- You are ready to run the scorer_csv from console by using:
```python src/score_csv.py --model model/model.pkl --manifest model/manifest.json --input_csv data/blind_test_data.csv```


## Docker
You can also use docker to generate a new prediction csv file, 

first you need to build the docker image: 

```docker build -t mlt-regression-scorer:dev -f docker/Dockerfile .```

Then, you can generate the predictions file using the next command:

```docker run --rm  -v "${PWD}\data:/app/data"  -v "${PWD}\model:/app/model"  -v "${PWD}\out:/app/out"  mlt-regression-scorer:dev  --model /app/model/model.pkl  --manifest /app/model/manifest.json  --input_csv /app/data/blind_test_data.csv  --output_csv /app/out/out.csv```

## Tests

This repo includes a focused test to ensure `score_csv.py` fails accordingly when the input CSV is missing any required feature (as defined in `model/manifest.json`).

### How to run
1. Activate your virtual environment.
2. Install pytest: `python -m pip install pytest`
3. From the repo root, run:

```bash
python -m pytest -q test/test_scorer_csv.py