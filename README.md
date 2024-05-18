## TBRD-null-pipeline
Repo to store the temporary minimal solution for alerting worrying sleep signs in out TBRD patients


### Installation
```
git clone git@github.com:BCM-Neurosurgery/TRBD-null-pipeline.git
pip install -e .
```

### Input Dir Structure
- oura
    - Percept004
        - 2023-06-22 # date of saved data
        - 2023-06-23
        - 2023-06-24
        - 2023-06-27
    - Percept005
        - 2023-06-22
        - 2023-06-23
        - 2023-06-24
        - 2023-06-27


### Output Dir Structure
- oura_out
    - Percept004
        - 2023-06-22 # date when program is run
        - 2023-06-23
        - 2023-06-24
        - 2023-06-27
    - Percept005
        - 2023-06-22
        - 2023-06-23
        - 2023-06-24
        - 2023-06-27


### TODOs
- send one email with all patients