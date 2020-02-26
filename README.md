# FEUP-DISS

Dissertation topic on a **Software Library for Stream Based Recommender Systems**.

## Communication Diagram

![communication](./images/communication.png)

1. Data Streams can be created through heir own class, or through file loader or stream generator methods.
2. They are sent to a prequential evaluator, so that we can test the current model before the updating phase.
3. The model is updated with the data stream.
4. The model is created through incremental algorithms.

## Interaction Diagram

![interaction](./images/interaction.png)

This diagram describes the process done whenever a new data stream
is provided by the user. If no new data stream arrives, then obviously, the model update part does not happen.

## Dependencies
Pipenv

## Tests
`pipenv run python3 -m unittest discover -s src -p "*_test.py"`