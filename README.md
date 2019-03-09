# Places365-server
A lightweight server for the Places365 scene classification models, using pytorch 0.4 and starlette.
See http://places2.csail.mit.edu/ and https://github.com/CSAILVision/places365 for the models. 

## Setup
1) Install dependencies via conda:
```conda env create -f environment.yml```
2) Activate environment: ```source activate places365```
3) Specify model architecture, host and port in `serve_places365.py` and run it.
4) Run `example_query.py` to classify an example image.

Example output: {'patio': 0.621, 'restaurant_patio': 0.296, 'porch': 0.021, 'beer_garden': 0.018, 'courtyard': 0.012}
