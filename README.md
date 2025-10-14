# multivariate-regression-batch-api
Production-ready multivariate regression: batch scoring


# Docker
You can use docker to generate new prediction csv file, 

first you need to build the docker image:
`docker build -t mlt-regression-scorer:dev -f docker/Dockerfile .`

Then you can generate the predictions file using the next command:
`docker run --rm  -v "${PWD}\data:/app/data"  -v "${PWD}\model:/app/model"  -v "${PWD}\out:/app/out"  mlt-regression-scorer:dev  --model /app/model/model.pkl  --manifest /app/model/manifest.json  --input_csv /app/data/blind_test_data.csv  --output_csv /app/out/out.csv`

## Test
This repo comes with a test to validate that the scorer_csv is behaving correctly when a dataframe with missing features is send, to run the test use:
`python -m pytest -q`