# multivariate-regression-batch

This repository shows an implementation in batch for a multivariate regression model already trained and persisted in `model/model.pkl`

To generate predictions using this model please follow the instructions:

## Requirements
- After cloning the repo, create a virtual environment using: 
```bash
py -3.11 -m venv .venv
```

- Activate the environment
```bash
.\.venv\Scripts\Activate.ps1
```
or in Bash/WSL:
```bash
source .venv/bin/activate
```

- Install required libraries:
```bash
python -m pip install -r requirements.txt
```

- As the results are going to be stored in the folder 'out' make sure it exists before running the scorer_csv file. Once ready, type in console:
```bash
python src/score_csv.py --model model/model.pkl --manifest model/manifest.json --input_csv data/blind_test_data.csv --output_csv out/target_pred.csv
```


## Docker
You can also use docker to generate a new prediction csv file, 

first you need to build the docker image: 

```bash
docker build -t mlt-regression-scorer:dev -f docker/Dockerfile .
```

Then, you can generate the predictions file using the next command:

```bash
docker run --rm  -v "${PWD}\data:/app/data"  -v "${PWD}\model:/app/model"  -v "${PWD}\out:/app/out"  mlt-regression-scorer:dev  --model /app/model/model.pkl  --manifest /app/model/manifest.json  --input_csv /app/data/blind_test_data.csv  --output_csv /app/out/target_pred.csv
```

## Tests

This repo includes a focused test to ensure `score_csv.py` fails accordingly when the input CSV is missing any required feature (as defined in `model/manifest.json`).

### How to run
1. Activate your virtual environment.
2. Install pytest: `python -m pip install pytest`
3. From the repo root, run:

```bash
python -m pytest -q tests/test_scorer_csv.py
```