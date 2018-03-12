# FakeNewsIdentification
Fake News Identification, part of the Big Data Submodule at Durham University.

## Running Instructions:
To run the code, first install the dependencies. We use Pipenv to do this, which outlines the requirements.

Install [pipenv](http://pipenv.readthedocs.io/en/latest/) and then run:

```bash
pipenv install
```

Now that's complete, open the virtualenv shell by running:

```bash
pipenv shell
```


### Running Application
Now that the above steps have been taken, run the shallow classifier using:

```
python3 main.py --shallow
```

Run the deep classifier using:

```
python3 main.py --deep
```
