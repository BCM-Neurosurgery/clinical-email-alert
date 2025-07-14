## TBRD-null-pipeline
Pathfinder of clinical email monoriring and patient state alert system 

## Citation

```
Tomasz M. Frączek, Yewen Zhou, Thomas P. Kutcher, Raphael A. Bechtold, Saipravallika Chamarthi, Nora Vanegas Arroyave, Wayne K. Goodman, Sameer A. Sheth, Jeffrey A. Herron, Nicole R. Provenza, 
Automated Monitoring and Clinical Notifications of Patient State for Neuropsychiatric Neuromodulation Studies, IEEE EMBC 2025 Proceedings
```

### Deploy to Elias
```
ssh auto@10.18.7.74
cd CODE
git clone git@github.com:BCM-Neurosurgery/TRBD-null-pipeline.git
source ~/miniconda3/bin/activate
conda activate trbdv0
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
