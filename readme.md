# bo-serving

This package provides a way to serve a Bayesian Optimiztion model as an HTTP endpoint. It was developed closely with the [Pozzo Research Group color matching demonstration](https://github.com/pozzo-research-group/jubilee_pipette_BOdemo/tree/main) for use in autonomous experimentation applications. It is still in early development. This package uses flask to provide an endpoint for a BoTorch or Ax model. 

If you are here from the Jubilee color match demonstration documentation, you want to follow these steps:

1. Clone this repository 
2. Install this package into the same environment that you have `science_jubilee` and `jubilee_pipette_BOdemo` installed into (i.e. `pip install -e .`)
3. cd into the `src/bo_serving/botorch_serving` directory
4. Spin up the flask app with `flask --app serve_botorch run`
5. Return to the color match demonstration documentation 