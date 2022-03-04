# Multiple-choice machine reading comprehension
Machine Reading Comprehension on the ReClor dataset. Additional evaluation is performed on RACE and COSMOSQA.

## Dependencies
pip install torch <br />
pip install keras <br />
pip install datasets <br />
pip install transformers <br />

## Navigation

By default, the provided scripts are for training and evaluation on ReClor within `electra_multipleChoice` when using the ELECTRA model or `albert_multipleChoice` when using the ALBERT model. Equivalent scripts are provided for the RACE and COSMOSQA within `race` and `cosmosqa` respectively. Further scripts are provided to handle the situation of when there is unanswerability present in the dataset (see paper - coming soon).

